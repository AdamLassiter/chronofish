# Rules

## Dimensions

To understand the game it is important to understand how the board operates. The board consists of 4 playable dimensions:

* File (x): This is left and right inside a board, just as in regular chess.
* Rank (y): This is up and down inside a board, just as in regular chess.
* Turn (T): This is moving to the past or to the future. This can be seen as moving to boards on the left or to the right.
* Timeline (L): This is moving between different timelines. This can be seen as moving to boards up or down.

Of these four dimensions, x and y are spatial dimensions, while T and L are temporal dimensions.

Spatial dimensions are constrained to a board itself. For example, an 8x8 board remains 8x8 regardless of how many turns have been played.

Temporal dimensions grow as moves are made. As the game progresses, the number of boards in the past increases. A given (T, L) coordinate corresponds to two boards: one for White's turn and one for Black's.

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

## Timelines

When a player moves a piece to a historical board (there are boards after it in the same timeline), a new timeline is added above or below all current timelines.
New timelines are created adjacent to the current ones.
Timelines created by White extend in one direction, whereas timelines created by Black extend in the opposite direction.

Each player is allowed to create an arbitrary number of timelines.
However, if one player creates two or more timelines than the other, the extra timelines will be inactive.
Such timelines are functionally optional, as shifting the Present does not demand any interaction with them.

### Inactive Timelines

Timelines are inactive when the player that created them has branched too much.
Boards on inactive timelines (referred to as "inactive boards") do not affect the location of the Present and thus, usually no moves need be made on them.
The existence of inactive timelines is a game balance concession to prevent one player from continuously branching, and also causes having timeline advantage (having less timelines created than your opponent) to be worth slightly more than a queen.

## Check

Check occurs when a royal piece (a king or a royal queen) is under threat of capture.
Check can be moved into (in a way that would allow the opponent to capture the king on their next move), but the game does not allow the submission of moves that leave the king in check.

### Checkmate

Mate occurs when there is no sequence of legal moves that leaves the Present on the opponent's color and none of the active player's kings (or royal queens) in check. If one of the active player's royal pieces is in check, it is checkmate; otherwise, it is stalemate.
