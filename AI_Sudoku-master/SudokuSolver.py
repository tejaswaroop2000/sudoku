from operator import attrgetter


class Solver:
    def checkvalidpuzzle(self, arr):
        subsquarestartingpoints = [[0, 0], [0, 3], [0, 6], [3, 0], [3, 3], [3, 6], [6, 0], [6, 3], [6, 6]]
        for row in range(9):
            has = set()
            for col in range(9):
                if arr[row][col] == 0:
                    continue
                if arr[row][col] in has:
                    return False
                has.add(arr[row][col])
        for col in range(9):
            has = set()
            for row in range(9):
                if arr[row][col] == 0:
                    continue
                if arr[row][col] in has:
                    return False
                has.add(arr[row][col])
        for pointrow, pointcol in subsquarestartingpoints:
            has = set()
            for row in range(3):
                for col in range(3):
                    if arr[pointrow+row][pointcol+col] == 0:
                        continue
                    if arr[pointrow+row][pointcol+col] in has:
                        return False
                    has.add(arr[pointrow+row][pointcol+col])
        return True

    def print_board(self, arr):
        for i in range(9):
            for j in range(9):
                if arr[i][j]==0:
                    print("_", end=" ")
                else:
                    print(arr[i][j], end=" ")
            print("")

    @staticmethod
    def solve_sudoku(arr):
        positions = []

        def add_position(ch, r, c, x):
            positions.append([ch, [
                9 * r + x,
                81 + 9 * c + x, 
                162 + 9 * ((r // 3) * 3 + (c // 3)) + x,
                243 + 9 * r + c
            ]])

        choice_row = 0
        for i in range(9): 
            for j in range(9):  
                if arr[i][j] == 0:
                    for k in range(9):  
                        add_position(choice_row, i, j, k)
                        choice_row += 1
                else:
                    k = arr[i][j] - 1
                    add_position(choice_row + k, i, j, k)
                    choice_row += 9
        alg_x = AlgorithmX(324, positions)
        if not alg_x.solve():
            return False
        rows = alg_x.solution
        if len(rows) != 81:
            return False
        for row in rows:
            i, row = divmod(row, 81)
            j, value = divmod(row, 9)
            arr[i][j] = value + 1 
        return True


class AlgorithmXNode:
    def __init__(self, value=0):
        
        self.value = value
        self.left = self.right = self.up = self.down = self.top = self

    def insert_h(self):
        self.left.right = self.right.left = self

    def insert_v(self, update_top=True):
        self.up.down = self.down.up = self
        if update_top:
            self.top.value += 1

    def insert_above(self, node):
        self.top = node.top
        self.up = node.up
        self.down = node
        self.insert_v()

    def insert_after(self, node):
        self.right = node.right
        self.left = node
        self.insert_h()

    def remove_h(self):
        self.left.right = self.right
        self.right.left = self.left

    def remove_v(self, update_top=True):
        self.up.down = self.down
        self.down.up = self.up
        if update_top:
            self.top.value -= 1

    def cover(self):
        self.top.remove_h()
        for row in self.top.loop('down'):
            for node in row.loop('right'):
                node.remove_v()

    def uncover(self):
        for row in self.top.loop('up'):
            for node in row.loop('left'):
                node.insert_v()
        self.top.insert_h()

    def loop(self, direction):
        
        if direction not in {'left', 'right', 'up', 'down'}:
            raise ValueError(f"Direction must be one of 'left', 'right', 'up', 'down', got {direction}")
        next_node = attrgetter(direction)
        node = next_node(self)
        while node != self:
            yield node
            node = next_node(node)


class AlgorithmX:
   
    def __init__(self, constraint_count, matrix):
        matrix.sort()
        headers = [AlgorithmXNode() for _ in range(constraint_count)]
        for row, cols in matrix:
            first = None  
            for col in cols:
                node = AlgorithmXNode(row)
                node.insert_above(headers[col])
                if first is None:
                    first = node
                else:
                    node.insert_after(first)
        
        self.root = AlgorithmXNode()
        last = self.root
        for header in headers:
            header.insert_after(last)
            last = header
        self.solution = []

    def solve(self):
        if self.root.right == self.root:
            
            return True
        
        header = min(self.root.loop('right'), key=attrgetter('value'))
        if header.value == 0:
           
            return False
        header.cover()
        for row in header.loop('down'):
            for node in row.loop('right'):
                node.cover()
            if self.solve():
                
                self.solution.append(row.value)
                return True
            for node in row.loop('left'):
                node.uncover()
        header.uncover()
        return False
