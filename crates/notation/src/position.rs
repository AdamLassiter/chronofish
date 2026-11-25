//! 5DFEN setup and 5DUCI position histories.

use chronofish_5d_rules::core::{
    action::{Action, ExtMove},
    multiverse::TimelineLayout,
    promotion::{PromotionOptions, format_promotions, parse_promotions},
    state::State,
    variants::{BoardSpec, Sign, VariantSetup, startpos, state_from_setup},
    vec4::Vec4,
};

/// Parse one `fen:l:t:c` block (without brackets); `l` may carry a +/- sign.
pub fn parse_board_fen_metadata(s: &str) -> Result<BoardSpec, String> {
    let mut parts = s.splitn(4, ':');
    let fen = parts.next().unwrap_or("").to_string();
    let ls = parts
        .next()
        .ok_or_else(|| format!("Expect line in board string:{}", s))?;
    let ts = parts
        .next()
        .ok_or_else(|| format!("Expect time in board string:{}", s))?;
    let cs = parts
        .next()
        .ok_or_else(|| format!("Expect color in board string:{}", s))?;
    let (sign, lnum) = if let Some(r) = ls.strip_prefix('+') {
        (Sign::Positive, r)
    } else if let Some(r) = ls.strip_prefix('-') {
        (Sign::Negative, r)
    } else {
        (Sign::Nil, ls)
    };
    let l: i32 = lnum
        .trim()
        .parse()
        .map_err(|_| format!("Expect number after ':': {}", s))?;
    let t: i32 = ts
        .trim()
        .parse()
        .map_err(|_| format!("Expect number after ':': {}", s))?;
    if !(0..=Vec4::L_MAX).contains(&l) || !(0..=Vec4::T_MAX).contains(&t) {
        return Err(format!("Board coordinate out of range: {}", s));
    }
    let c = match cs.trim() {
        "w" | "W" => false,
        "b" | "B" => true,
        _ => return Err(format!("Unknown color in {}", s)),
    };
    Ok(BoardSpec { fen, sign, l, t, c })
}

/// Build a state from a 5DUCI `position` argument string (`startpos` or
/// `[size MxN [odd|even]] fen [..][..]`).
pub fn position_from_uci(position: &str) -> Result<State, String> {
    let position = position.trim();
    if position == "startpos" {
        return Ok(startpos());
    }
    let (mut m, mut n, mut even) = (0, 0, false);
    let mut promotions = PromotionOptions::QUEEN;
    let fen: &str;
    if let Some(rest) = position.strip_prefix("size ") {
        match rest.find(" fen ") {
            None => fen = rest,
            Some(fp) => {
                let mut it = rest[..fp].split_whitespace();
                if let Some(sz) = it.next()
                    && let Some((a, b)) = sz.split_once('x')
                {
                    m = a.parse().unwrap_or(0);
                    n = b.parse().unwrap_or(0);
                }
                for tok in it.by_ref() {
                    match tok {
                        "even" => even = true,
                        "promotions" => break,
                        _ => {}
                    }
                }
                // `promotions <pieces>`: the pieces pawns promote to (as 5dpgn's header: letters,
                // or `*`); without it, the queen only.
                if let Some(p) = it.next() {
                    promotions = parse_promotions(p)?;
                }
                fen = &rest[fp + 5..];
            }
        }
    } else if let Some(rest) = position.strip_prefix("fen ") {
        fen = rest;
    } else if let Some(rest) = position.strip_prefix("even fen ") {
        even = true;
        fen = rest;
    } else if let Some(rest) = position.strip_prefix("odd fen ") {
        fen = rest;
    } else {
        fen = position;
    }
    let mut boards = Vec::new();
    let bytes = fen.as_bytes();
    let mut pos = 0;
    while pos < bytes.len() {
        while pos < bytes.len() && bytes[pos].is_ascii_whitespace() {
            pos += 1;
        }
        if pos >= bytes.len() {
            break;
        }
        if bytes[pos] == b'[' {
            match fen[pos..].find(']') {
                None => {
                    boards.push(parse_board_fen_metadata(&fen[pos..])?);
                    break;
                }
                Some(e) => {
                    boards.push(parse_board_fen_metadata(&fen[pos + 1..pos + e])?);
                    pos += e + 1;
                }
            }
        } else {
            boards.push(parse_board_fen_metadata(&fen[pos..])?);
            break;
        }
    }
    state_from_setup(&VariantSetup {
        size_x: if m > 0 { m } else { 8 },
        size_y: if n > 0 { n } else { 8 },
        boards,
        even,
        promotions,
    })
}

/// Apply a 5DUCI move history (`<lan>`/`submit` tokens) to `s`.
pub fn apply_uci_moves(s: &mut State, moves: &str) -> Result<(), String> {
    for tok in moves.split_whitespace() {
        if tok == "submit" {
            if !s.submit::<false>() {
                return Err("position error: cannot submit move history".into());
            }
        } else {
            let mv = ExtMove::parse(tok)
                .map_err(|e| format!("position error: cannot parse move {}: {}", tok, e))?;
            match s.normalize_promotion(mv) {
                Some(n) if s.apply_ext::<false>(n) => {}
                _ => return Err(format!("position error: cannot apply move {}", tok)),
            }
        }
    }
    Ok(())
}

/// Parse a "G <promotions> <5duci position> moves <lan|submit>*" line.
pub fn parse_g_line(line: &str) -> Result<State, String> {
    let rest = line.strip_prefix("G ").ok_or("not a G line")?;
    let (promo, rest) = rest.split_once(' ').ok_or("bad G line")?;
    let (pos, moves) = match rest.find(" moves") {
        Some(i) => (&rest[..i], &rest[i + 6..]),
        None => (rest, ""),
    };
    let mut s = position_from_uci(pos)?;
    s.set_promotion_options(PromotionOptions(
        promo.parse::<u8>().map_err(|e| e.to_string())?,
    ));
    apply_uci_moves(&mut s, moves)?;
    Ok(s)
}

/// Replay a reference game history, checking royal safety at every submission.
///
/// A trailing draft is permitted. The original state is preserved on error.
pub fn replay_checked_uci(initial: &State, moves: &str) -> Result<State, String> {
    use chronofish_5d_rules::core::referee::{apply_draft_move, submit_draft};

    let mut state = initial.clone();
    for token in moves.split_whitespace() {
        if token == "submit" {
            state = submit_draft(&state)?;
        } else {
            apply_draft_move(&mut state, ExtMove::parse(token)?)?;
        }
    }
    Ok(state)
}

/// Parse a list of LAN tokens (`submit` tokens are ignored).
pub fn parse_lan_moves(text: &str) -> Result<Vec<ExtMove>, String> {
    text.split_whitespace()
        .filter(|t| *t != "submit")
        .map(ExtMove::parse)
        .collect()
}

/// The action's LAN tokens, played in order from `s`.
pub fn action_lan(s: &State, act: &Action) -> String {
    let mut os = s.clone();
    let mut parts = Vec::new();
    for em in act.moves() {
        parts.push(em.lan(&os));
        if let Some(n) = os.normalize_promotion(*em) {
            os.apply_ext::<false>(n);
        }
    }
    parts.join(" ")
}

/// The 5DUCI `position` argument for a setup.
pub fn setup_to_uci(v: &VariantSetup) -> String {
    let mut s = format!(
        "size {}x{} {}",
        v.size_x,
        v.size_y,
        if v.even { "even" } else { "odd" }
    );
    if v.promotions != PromotionOptions::QUEEN {
        s += &format!(" promotions {}", format_promotions(v.promotions));
    }
    s += " fen";
    let layout = if v.even {
        TimelineLayout::even()
    } else {
        TimelineLayout::odd()
    };
    for b in &v.boards {
        // Internal negative lines are written mirrored with a '-' sign.
        let (sign, l) = match b.sign {
            Sign::Nil if b.l < 0 => ("-", layout.mirror(b.l)),
            Sign::Nil => ("", b.l),
            Sign::Positive => ("+", b.l),
            Sign::Negative => ("-", b.l),
        };
        s += &format!(
            " [{}:{}{}:{}:{}]",
            b.fen,
            sign,
            l,
            b.t,
            if b.c { "b" } else { "w" }
        );
    }
    s
}
