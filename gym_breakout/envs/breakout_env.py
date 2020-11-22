import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np


rows = 6
cols = 6
bar_initial_pos = 1
possible_actions = {"left": -1, "right": 1, "none": 0}
ball_initial_pos = (2,0)
ball_initial_direction =(1,-1)
max_reward = 100

class breakoutEnv(gym.Env):
    def __init__(self):
        self._rows = rows
        self._cols = cols

        self._bar = bar_initial_pos
        self._ball_pos = ball_initial_pos
        self._ball_dir = ball_initial_direction
        self._ball_possible_direction = [(1,-1),(1,1),(-1,-1),(-1,1)]
        self.max_reward = max_reward
        self._bricks = tuple(np.full(cols, True).tolist())
        self.initialState = (self._bar, (self._ball_pos, self._ball_dir), self._bricks)
        
        self.actionsName = list(possible_actions.keys())
        self.actions = possible_actions
        self.actionNumber = len(self.actions)

        self.S = self.genereteAllStates()                       #S contiene tutti gli stati in cui è possibile trovarsi
        self.stateNumber = len(self.S)
        self.P = self.createP()                                 #P: Transiction Model, type dictonary{state : dictonary{action : Tuple(nextState, reward, done)}}
        
        
        '''
        A state is a Tuple (bar, (posB,dirB), bricks) where:
        bar = position of the bar (the bar is at (0,bar) in the game's grid)
        posB = a pair that indicate the position of the ball in the game's grid
        dirB = a pair that indicate the direction of the ball in the game's grid
        bricks = a List that represents where are the bricks at the final row. grid[i] = True if in the cell (rows-1,i) there is a wall, False if it is free
        '''

    def step(self, a):
        reward = 0.0
        done = False
        info = {}
        action = self.to_action(a)

        reward, done = self.compute_action(action)

        ob = (self._bar, (self._ball_pos, self._ball_dir), self._bricks)

        return ob, reward, done, info


    def to_action(self, a):
            if(not (a in possible_actions)):
                return possible_actions["none"]
            else:
                return possible_actions[a]

    def compute_action(self, action):
        reward = 0.0
        done = False

        new_bar_position = self._bar + action                       #Muovi la barra nella nuova posizione
        if(new_bar_position>=0 and new_bar_position<self._cols):    #Se la nuova posizione è nella griglia...
            self._bar = new_bar_position                                #...aggiorna la posizione della pallina alla nuova

        self._move_ball()

        if(self._ball_pos[0] == 0):                                 #Se la pallina sta a livello della barra, o sta sulla sbarra o hai perso...
            if(self._ball_pos[1] is self._bar):                         #...Se la pallina colpisce la barra...
                reward = 1
                self._change_direction()                                    #...Inverti direzione y della pallina
                return reward, done
            else:                                                   #Se non c'è la barra...
                reward = -100.0
                done = True
                return reward, done                                     #...GameOver con reward negativa

        (r,d) = self._ball_pos
        if(self._ball_pos[0] == self._rows-1):                      #Se stai sull'ultima riga (dove stanno i blocchi)...
            if(self._bricks[d]):                                        #...Se hai colpito un blocco
                listBrick = list(self._bricks)
                listBrick[d] = False                                        #...Distruggi il blocco
                self._bricks = tuple(listBrick)
                reward = 1.0                                                #...reward positiva
                done = self._is_done()                                      #...Controlla se hai distrutto tutti i blocchetti
                if done:                                                    #...Se era l'ultimo 
                    reward = self.max_reward                                    #...massima reward, hai vinto!
                self._change_direction()                                    #...inverti direzione y della pallina
        return reward, done



    def _move_ball(self):
        (ri,ci) = self._ball_pos
        (rd,cd) = self._ball_dir

        rf = ri + cd
        if(rf>=self._rows):
            rf = ri
            cd = -cd

        cf = ci + rd
        if(cf>=self._cols or cf<0):
            cf = ci
            rd = -rd
        self._ball_pos = (rf,cf)
        self._ball_dir = (rd,cd)

    def _change_direction(self):
        (rd,cd) = self._ball_dir
        self._ball_dir = (rd,-cd)

    def _is_done(self):
        return not (True in self._bricks)               #Se nei bricks sono tutti False (free) allora hai distrutto tutti i brick, torna True

    def reset(self):
        print("RESET!")
        self._bar = bar_initial_pos
        self._ball_pos = ball_initial_pos
        self._ball_dir = ball_initial_direction

        self._bricks = tuple(np.full(cols, True).tolist())
        return


    def render(self, mode='human'):
        (rb,cb) = self._ball_pos
        grid = np.zeros((self._rows, self._cols))
        for i, val in enumerate(self._bricks):
            grid[self._rows - 1][i] = 1 if val else 0
        grid[rb,cb] = 2
        y = self._cols
        x = self._rows-1
        first_line = "--"
        while y>0:
            first_line += "-"
            y -= 1
        print(first_line)
        while(x>0):
            self._write_line(grid[x])
            x -= 1
        last_line = "|"
        y=0
        while y<self._rows:
            if y==self._bar:
                word = '±' if grid[0][y] == 2 else '_'
            else:
                word = '+' if grid[0][y] == 2 else ' '
            last_line += word
            y += 1
        last_line += "|"
        print(last_line)
        return

    def _write_line(self, row):
        line = '|'
        y=0
        while y<self._rows:
            word = ' ' if row[y] == 0 else '@' if row[y] == 1 else '+'
            line += word
            y += 1
        line += '|'
        print(line)

    def close(self):
        pass
    
    def _changeState(self, state):
        self._bar = state[0]
        self._ball_pos = state[1][0]
        self._ball_dir = state[1][1]
        self._bricks = state[2]
    
    '''
    nextStep give an initial state and a action returns the new state given by the esecution of the action from the state,
    the reward and a boolean done.
    '''
    def nextState(self, state, action):
        temp = (self._bar, (self._ball_pos, self._ball_dir), self._bricks)
        self._changeState(state)
        ob, reward, done, info = self.step(action)
        self._changeState(temp)
        return ob, reward, done
        
    def allPossibleBrick(self, i):
        if i==1:
            return [[False],[True]]
        result1 = self.allPossibleBrick(i-1)
        result = []
        for l in result1:
            l1 = l.copy()
            l1.append(True)
            l.append(False)
            result.append(l1)
            result.append(l)
        return result
    
    def possibleBrick(self, i):
        l = self.allPossibleBrick(i)
        result = []
        for x in l:
            result.append(tuple(x))
        result.remove(tuple(np.full(5, False)))
        return result   
    
    def genereteAllStates(self):
        return self.generateAllStatesRecursive(self.initialState, [])
        
    def generateAllStatesRecursive(self, state, S):
        S.append(state)
        for action in self.actions.keys():
            nextState, reward, done = self.nextState(state, action)
            if not (done or nextState in S):
                S = self.generateAllStatesRecursive(nextState, S)
        return S
 
    #Il generatore di stati sotto non prevede la visita come quello di sopra.
    #Quello con la visita genera ovviamente meno stati, solo quelli realmente raggiungibili. Quello sotto è comunque funzionante.
    '''
    def genereteAllStates(self):
        A = []
        for bar in range(self._rows):
            for ball_row in range(self._rows):
                for ball_col in range(self._cols):
                    for (dir_r, dir_c) in self._ball_possible_direction:
                        for walls in self.possibleBrick(self._rows):
                            valid = True
                            if ball_row == self._rows-1:
                                valid = not walls[ball_col]
                            if ball_row == 0:
                                if ball_col != bar:
                                    valid = False
                                else:
                                    valid = dir_c > 0
                            if valid:
                                state = (bar, ((ball_row, ball_col),(dir_r, dir_c)), walls)
                                A.append(state)
        return A
    '''
    
    def createP(self):
        P = {}
        for state in self.S:
            nexts = {}
            for action in self.actions.keys():
                nextState, reward, done = self.nextState(state, action)
                nexts[action] = (nextState, reward, done)
            P[state] = nexts
        return P