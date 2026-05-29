struct ParsedMove {
    from: Position,
    to: Position,
    piece: char,
    captured: Option<char>,
    branch_timeline_id: Option<i32>,
}

impl ParsedMove {
    fn parse(text: &str) -> Result<Self, String> {
        let chars: Vec<char> = text.trim().chars().collect();
        let mut index = 0;
        let from_time = parse_prefixed_i32(&chars, &mut index, 'T')?;
        let from_timeline = parse_prefixed_i32(&chars, &mut index, 'L')?;
        let (from_x, from_y) = parse_square(&chars, &mut index)?;
        let piece = *chars
            .get(index)
            .ok_or_else(|| format!("Missing piece in `{text}`"))?;
        if piece_type_from_notation(piece).is_none() {
            return Err(format!("Unknown piece `{piece}` in `{text}`"));
        }
        index += 1;

        let (to_time, to_timeline) = if chars.get(index) == Some(&'T') {
            let time = parse_prefixed_i32(&chars, &mut index, 'T')?;
            let timeline = parse_prefixed_i32(&chars, &mut index, 'L')?;
            (time, timeline)
        } else {
            (from_time, from_timeline)
        };
        let (to_x, to_y) = parse_square(&chars, &mut index)?;

        let captured = if chars.get(index) == Some(&'x') {
            index += 1;
            let captured = *chars
                .get(index)
                .ok_or_else(|| format!("Missing captured piece in `{text}`"))?;
            if piece_type_from_notation(captured).is_none() {
                return Err(format!("Unknown captured piece `{captured}` in `{text}`"));
            }
            index += 1;
            Some(captured)
        } else {
            None
        };

        let branch_timeline_id = if chars.get(index) == Some(&'>') {
            index += 1;
            Some(parse_prefixed_i32(&chars, &mut index, 'L')?)
        } else {
            None
        };

        if matches!(chars.get(index), Some('+') | Some('#')) {
            index += 1;
        }
        if index != chars.len() {
            return Err(format!("Unexpected suffix in `{text}`"));
        }

        Ok(Self {
            from: Position {
                timeline_id: from_timeline,
                time: from_time,
                x: from_x,
                y: from_y,
            },
            to: Position {
                timeline_id: to_timeline,
                time: to_time,
                x: to_x,
                y: to_y,
            },
            piece,
            captured,
            branch_timeline_id,
        })
    }
}

fn strip_notation_comments(line: &str) -> String {
    let mut output = String::new();
    let mut in_comment = false;
    for character in line.chars() {
        match character {
            '[' => in_comment = true,
            ']' => in_comment = false,
            _ if !in_comment => output.push(character),
            _ => {}
        }
    }
    output
}

fn parse_prefixed_i32(chars: &[char], index: &mut usize, prefix: char) -> Result<i32, String> {
    if chars.get(*index) != Some(&prefix) {
        return Err(format!("Expected `{prefix}`"));
    }
    *index += 1;
    let start = *index;
    if chars.get(*index) == Some(&'-') {
        *index += 1;
    }
    while chars.get(*index).is_some_and(|character| character.is_ascii_digit()) {
        *index += 1;
    }
    if *index == start || *index == start + 1 && chars.get(start) == Some(&'-') {
        return Err(format!("Expected number after `{prefix}`"));
    }
    chars[start..*index]
        .iter()
        .collect::<String>()
        .parse::<i32>()
        .map_err(|error| error.to_string())
}

fn parse_square(chars: &[char], index: &mut usize) -> Result<(i32, i32), String> {
    let file = *chars.get(*index).ok_or_else(|| "Missing file".to_string())?;
    *index += 1;
    let rank = *chars.get(*index).ok_or_else(|| "Missing rank".to_string())?;
    *index += 1;
    let x = match file {
        'a' => 0,
        'b' => 1,
        'c' => 2,
        'd' => 3,
        'e' => 4,
        'f' => 5,
        'g' => 6,
        'h' => 7,
        _ => return Err(format!("Invalid file `{file}`")),
    };
    let y = rank
        .to_digit(10)
        .filter(|rank| (1..=8).contains(rank))
        .ok_or_else(|| format!("Invalid rank `{rank}`"))? as i32
        - 1;
    Ok((x, y))
}

fn piece_type_from_notation(symbol: char) -> Option<PieceType> {
    match symbol.to_ascii_uppercase() {
        'K' => Some(PieceType::King),
        'N' => Some(PieceType::Knight),
        'B' => Some(PieceType::Bishop),
        'R' => Some(PieceType::Rook),
        'Q' => Some(PieceType::Queen),
        'P' => Some(PieceType::Pawn),
        'U' => Some(PieceType::Unicorn),
        'D' => Some(PieceType::Dragon),
        'S' => Some(PieceType::Princess),
        'W' => Some(PieceType::Brawn),
        'C' => Some(PieceType::CommonKing),
        'Y' => Some(PieceType::RoyalQueen),
        _ => None,
    }
}
