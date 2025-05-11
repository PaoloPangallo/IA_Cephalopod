import random
from mAIN.utils.strategy_utils import get_opponent, get_all_legal_moves, simulate_move


class SmartLookaheadStrategy5:
    def opponent_can_capture_six(self, board, opponent_color):
        for move in get_all_legal_moves(board, opponent_color):
            _, top_face, _ = move
            if top_face == 6:
                return True
        return False

    def is_move_safe_after_lookahead(self, board, my_move, my_color):
        opponent = get_opponent(my_color)
        simulated_board = simulate_move(board, my_move, my_color)

        # Tutte le risposte legali dell'avversario
        opponent_moves = get_all_legal_moves(simulated_board, opponent)

        for response in opponent_moves:
            board_after_response = simulate_move(simulated_board, response, opponent)
            if self.opponent_can_capture_six(board_after_response, my_color):
                return False  # Dopo la risposta, l'avversario può fare 6
        return True

    def choose_move(self, game, state, player):
        legal_moves = game.actions(state)
        fallback_move = None

        # Primo turno: board vuota
        if all(cell is None for row in state.board for cell in row):
            r, c = random.choice([(r, c) for r in range(state.size) for c in range(state.size)])
            return ((r, c), 1, ())

        # 1. Prova catture buone (da 6 o ≤ 5), sicure anche dopo risposta
        for move in legal_moves:
            _, top_face, captured = move
            if captured and (top_face == 6 or top_face <= 5):
                if self.is_move_safe_after_lookahead(state, move, player):
                    return move

        # 2. Prova piazzamenti da 1 sicuri
        for move in legal_moves:
            _, top_face, captured = move
            if not captured and top_face == 1:
                if self.is_move_safe_after_lookahead(state, move, player):
                    return move
                if fallback_move is None:
                    fallback_move = move

        # 3. Fallback: meglio un 1 che niente
        if fallback_move:
            return fallback_move
        return random.choice(legal_moves)
