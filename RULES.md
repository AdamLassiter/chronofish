# Rules

## Rulesets and armies

Chronofish supports three deliberately separate rulesets:

* **Standard Chess** uses ordinary chess on one 8×8 board with the orthodox army.
* **5D Chess** uses the multiverse rules below with the orthodox army: rook, knight, bishop, queen, king, bishop, knight, rook, with pawns in front.
* **5D Variant Chess** uses princess, unicorn, dragon, royal queen, common king, dragon, unicorn, princess, with brawns in front.

In the variant army, the royal queen replaces the king as the piece that must be protected. The common king is not royal. A common king castles with a princess in the same spatial arrangement that an orthodox king castles with a rook.

## Dimensions

To understand the game it is important to understand how the board operates. The board consists of 4 playable dimensions:

* File (x): This is left and right inside a board, just as in regular chess.
* Rank (y): This is up and down inside a board, just as in regular chess.
* Turn (T): This is moving to the past or to the future. This can be seen as moving to boards on the left or to the right.
* Timeline (L): This is moving between different timelines. This can be seen as moving to boards up or down.

Of these four dimensions, x and y are spatial dimensions, while T and L are temporal dimensions.

Spatial dimensions are constrained to a board itself. For example, an 8x8 board remains 8x8 regardless of how many turns have been played.

Temporal dimensions grow as moves are made. As the game progresses, the number of boards in the past increases. Chronofish numbers every successive board: even `T` boards are White-to-move positions and odd `T` boards are Black-to-move positions. One unit of movement along the turn axis therefore changes the stored `T` coordinate by two and lands on another board of the moving colour.

## Pieces

### Pawn

The pawn moves one or two squares in the "forward" direction.
Here, "forward" includes the direction towards your opponent's back timeline (ie. away from timeline 0).
The pawn can attack forward one square along row/column (spacelike) and timeline/turn (timelike) diagonals.

### Brawn

The brawn is an alternative interpretation of the pawn that allows more capturing diagonals.
It can attack along any diagonal that includes a forward direction and does not include any backward direction.
This is another valid interpretation of the traditional pawn and has more available attacks than purely spacelike or purely timelike attacks.
The brawn is denoted by an upside-down pawn.

### Knight

The knight moves two squares in one dimension, then one square in a different dimension.
The knight uniquely does not need a clear path to make its move, and cannot be blocked or obstructed by other pieces.

### Unicorn

The unicorn can move any unobstructed distance along a triagonal (three dimensions simultaneously).
The unicorn is denoted by an upside-down knight.

### Bishop

The bishop moves any distance in two dimensions at once.

### Dragon

The dragon moves any number of unobstructed squares in all four dimensions at once (called a quadragonal).
The dragon is denoted by an upside-down bishop.

### Rook

The rook moves any distance in one dimension at a time. For example, while a rook is moving along a column, its row, turn and timeline are constant.

### Princess

The princess is an alternative interpretation of the queen that can only move in two dimensions at a time.
This means it is only rook and bishop movement, and not unicorn or dragon movement.
The princess is denoted by an upside-down rook.

### King

The king moves one square in any number of dimensions at once.

### Common King

The common king is a version of the king that is not royalty, meaning it does not have to be protected.
It otherwise moves the same as a king.
The common king is denoted by an upside-down king.

### Queen

Like in regular chess, the queen can move to any square that a rook or bishop on its square could move to. However, it is also extended to include the moves of the unicorn (triagonal) and the dragon (quadragonal).

### Royal Queen

The royal queen is a version of the queen that is royalty, meaning it has to be protected.
It otherwise moves the same as a queen.
The royal queen is denoted by an upside-down queen.

## Moving

On each playable board of your color (boards that don't have another board immediately after them in time), you may move one piece of your color. Doing so does not affect the original board; instead, it creates a new board immediately after, showing the result of the move.

If a piece is moved to a different board, either across time, timelines, or both, a move is considered to be made on *both* boards. On the new boards, the piece disappears from the board it moved from, and appears on the board it moved to.

If a piece is moved to a historical board (a board where a move has already been made), the game does not create a new board overlapping the original new board; instead, it creates that new board on a new timeline.

A player may have to make more than one move in a 5D turn. The turn can be submitted only when every required active board at the Present has advanced to the opponent's colour. Intermediate moves are staged until the whole turn is submitted.

The shared 5D reference engine decides which late or inactive fronts may
participate in a turn. A move can be staged only if a legal completed turn
contains that draft. An optional move can change how a later jump branches.

### Special moves

Both 5D modes use reference castling and queen promotion. The alternate mode
keeps its own starting army. Standard Chess retains orthodox special moves
and the four standard promotion choices.

## Timelines and check

The reference engine computes the active range and the Present. A complete
5D turn must advance the Present and protect all of the mover's royals.
A player with no legal turn loses by checkmate if checked. Otherwise the
reference stalemate and exiled-king rules apply. Standard Chess also
adjudicates threefold repetition, the fifty-move rule, and insufficient material.
