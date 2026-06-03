impl Game {
    fn load_notation(&mut self, notation: &str) -> Result<(), String> {
        let mut game = Game::new();
        for line in notation.lines() {
            let line = strip_notation_comments(line);
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let (_, turn_text) = line
                .split_once('.')
                .ok_or_else(|| format!("Missing turn prefix in `{line}`"))?;
            for move_text in turn_text.split('/') {
                let move_text = move_text.trim();
                if move_text.is_empty() {
                    continue;
                }
                game.apply_notation_move(move_text)?;
            }
            if game.submit_turn() == 0 {
                return Err(game.last_message.clone());
            }
        }

        *self = game;
        Ok(())
    }

    fn apply_notation_move(&mut self, move_text: &str) -> Result<(), String> {
        let parsed = ParsedMove::parse(move_text)?;
        let piece = self
            .piece_at(parsed.from)
            .ok_or_else(|| format!("No piece at source in `{move_text}`"))?;
        if piece.notation_symbol() != parsed.piece {
            return Err(format!(
                "Piece mismatch in `{move_text}`: source has {}",
                piece.notation_symbol()
            ));
        }

        let Some((_, move_kind)) = self.legal_move_kind(parsed.from, parsed.to) else {
            return Err(format!("Illegal move `{move_text}`"));
        };

        if let Some(captured) = parsed.captured {
            let actual = self
                .captured_piece(parsed.to, move_kind)
                .ok_or_else(|| format!("Capture marker without captured piece in `{move_text}`"))?;
            if actual.notation_symbol() != captured {
                return Err(format!(
                    "Capture mismatch in `{move_text}`: target has {}",
                    actual.notation_symbol()
                ));
            }
        }

        if let Some(actual_branch_timeline_id) = parsed.branch_timeline_id {
            let expected = next_branch_timeline_id(piece.color, self);
            if actual_branch_timeline_id != expected {
                return Err(format!(
                    "Branch timeline mismatch in `{move_text}`: expected L{expected}"
                ));
            }
        }

        if self.apply_move(parsed.from, parsed.to) == 0 {
            return Err(self.last_message.clone());
        }
        Ok(())
    }
}
