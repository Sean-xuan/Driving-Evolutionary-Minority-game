#%%Main Program
##Author Zixuan Xu, Lianghua Chen
import numpy as np
import random
from random import seed
import matplotlib.pyplot as plt
import copy

seed(12345)


class Evolution():
    def __init__(self, initgame_num, train_num, game_num, driver_num, oldrate=0.30, yourate=0.40, explrate=0.1,
                 sturate=0.1, updatrate= 0.01, error_mut = 0.1):
        self.num_drivers = driver_num
        self.new_mutant_errorrate = error_mut
        self.num_olddriver = int(oldrate * self.num_drivers)
        self.num_youdriver = int(yourate * self.num_drivers)
        self.num_explorer = int(explrate * self.num_drivers)
        self.num_stubbon = int(sturate * self.num_drivers)
        self.num_randomer = self.num_drivers - self.num_olddriver - self.num_youdriver - self.num_stubbon - self.num_explorer
        self.errorrate = 0.05
        # setting in the evolution game
        self.gamesnum = game_num
        self.update = 20
        self.updatenum = int(updatrate * self.num_drivers)
        # setting in the borad
        self.trainnum = train_num
        self.initgame_num = initgame_num
        self.m = 3
        self.darray = np.random.randint(0, 2, size=(2 ** self.m, 2 ** (2 ** self.m)))
        self.strategy = ['000', '001', '010', '011', '100', '101', '110', '111']

    def init_game(self):
        n = self.trainnum
        darray = np.random.randint(0, 2, size=(2 ** self.m, 2 ** (2 ** self.m)))
        virtual_score = np.zeros(2 ** (2 ** self.m))
        input_state = np.random.randint(0, 2, self.m)
        a = [str(i) for i in input_state]
        b = str(''.join(a))
        index1 = self.strategy.index(b)
        initial_agentchoice = random.sample(range(0, 2 ** (2 ** self.m)), n)
        slice = darray[index1:index1 + 1][0]
        round_decision = slice[initial_agentchoice].tolist()
        excess_demand = (round_decision.count(1) - round_decision.count(0)) / len(round_decision)
        virtual_score[initial_agentchoice] -= excess_demand * (2 * (slice[initial_agentchoice]) - 1)
        for time in range(self.initgame_num):
            before = input_state[-2:]
            if round_decision.count(1) > round_decision.count(0):
                input_state = np.append(before, 0)
            else:
                input_state = np.append(before, 1)
            a = [str(i) for i in input_state]
            b = str(''.join(a))
            index1 = self.strategy.index(b)
            score_choice = []
            round_decision = []
            round_score = []
            round_strategy = []
            for i in range(n):
                agent_choice = random.sample(range(0, 2 ** (2 ** self.m)), 2)
                if virtual_score[agent_choice[0]] > virtual_score[agent_choice[1]]:
                    score_choice.append(
                        [darray[index1][agent_choice[0]], virtual_score[agent_choice[1]], agent_choice[0]])
                else:
                    score_choice.append(
                        [darray[index1][agent_choice[0]], virtual_score[agent_choice[1]], agent_choice[1]])
                round_decision.append(score_choice[i][0])
                round_score.append(score_choice[i][1])
                round_strategy.append(score_choice[i][2])
            excess = (round_decision.count(1) - round_decision.count(0)) / len(round_decision)
            for g in round_strategy:
                ind = round_strategy.index(g)
                virtual_score[g] -= excess * (2 * (round_decision[ind]) - 1)
        self.darray = darray
        self.virtual_score = virtual_score

    def olddriver(self, olddrive_num):
        n = olddrive_num
        inputstate = self.input_state
        self.virtual_score
        a = [str(i) for i in inputstate]
        b = str(''.join(a))
        index1 = self.strategy.index(b)
        choices = []
        c=np.arange(0,self.num_drivers,1)
        d=self.attribute==0
        c=c[d==True]
        #for i in range(self.num_drivers):
        for i in c:
            if self.mutation[i]==0:
                agent_choice = random.sample(range(0, 2 ** (2 ** self.m)), 2)
                if self.virtual_score[agent_choice[0]] > self.virtual_score[agent_choice[1]]:
                    choices.append(self.darray[index1][agent_choice[0]])
                else:
                    choices.append(self.darray[index1][agent_choice[1]])
            elif self.mutation[i]==1:
                #mutation
                muta=copy.deepcopy(inputstate)
                temp=random.choice([0,1,2])
                muta[temp]=1-muta[temp]
                a = [str(i) for i in muta]
                index_muta = self.strategy.index(str(''.join(a)))
                agent_choice = random.sample(range(0, 2 ** (2 ** self.m)), 2)
                if self.virtual_score[agent_choice[0]] > self.virtual_score[agent_choice[1]]:
                    choices.append(self.darray[index_muta][agent_choice[0]])
                else:
                    choices.append(self.darray[index_muta][agent_choice[1]])
        self.olddriver_choice = choices

    def game(self):
        self.init_game()
        self.input_state = np.random.randint(0, 2, self.m)
        self.attribute = np.array( [0] * self.num_olddriver + [1] * self.num_youdriver + [2] * self.num_stubbon + [3] * self.num_randomer + [4] * self.num_explorer)
        self.error_attribute=np.array([0.05] * self.num_drivers)
        self.mutation=np.array([0] * self.num_olddriver+[2]*(self.num_drivers-self.num_olddriver))
        # for g in range(int(self.gamesnum/self.update)):
        res = np.zeros((5, self.gamesnum)).astype(int)
        for g in range(int(self.gamesnum)):
            self.scores = np.zeros(self.num_drivers)
            res[0][g], res[1][g], res[2][g], res[3][g], res[4][g] = len(np.where(self.attribute == 0)[0]), len(
                np.where(self.attribute == 1)[0]), len(np.where(self.attribute == 2)[0]), len(
                np.where(self.attribute == 3)[0]), len(np.where(self.attribute == 4)[0])
            #每20轮调整
            if g % 20 == 0: print(g, 'round', res[0][g], res[1][g], res[2][g], res[3][g], res[4][g])
            for i in range(self.update):
                self.roundchoice = np.zeros(self.num_drivers)
                self.roundchoice[self.attribute == 1] = self.input_state[-1]
                self.roundchoice[self.attribute == 2] = 0
                self.roundchoice[self.attribute == 4] = 1
                self.roundchoice[self.attribute == 3] = np.random.randint(0, 2, len(np.where(self.attribute == 3)[0]))
                self.olddriver(len(np.where(self.attribute == 0)[0]))
                self.roundchoice[self.attribute == 0] = self.olddriver_choice
                self.roundchoice = self.roundchoice.astype(int)
                #make error
                errors_random=np.random.uniform(0,1,self.num_drivers)
                self.roundchoice[errors_random<self.error_attribute]=1-self.roundchoice[errors_random<self.error_attribute]
                if list(self.roundchoice).count(0) > list(self.roundchoice).count(1):
                    self.scores += self.roundchoice
                    self.input_state = np.append(self.input_state[1:], 1)
                else:
                    self.scores += 1 - self.roundchoice
                    self.input_state = np.append(self.input_state[1:], 0)
            tmax = copy.deepcopy(self.scores)
            tmin = copy.deepcopy(self.scores)
            for _ in range(self.updatenum):
                number1 = max(tmax)
                best_driver = np.where(tmax == number1)[0][0]
                number2 = min(tmin)
                worst_driver = np.where(tmin == number2)[0][0]
                if number1-number2>3:
                    self.attribute[worst_driver] = self.attribute[best_driver]
                    #mutation to change state
                    if self.attribute[best_driver]==0:
                    # If mutant inherit old drivers' strategy
                        # self.mutation[worst_driver]=1
                        self.mutation[worst_driver]=0
                    self.error_attribute[worst_driver] = self.new_mutant_errorrate
                    # else:self.error_attribute[worst_driver] = self.new_mutant_errorrate
                    tmax[best_driver] = 0
                    tmin[worst_driver] = self.update
                else: print('stop')
        # step = [i for i in range(self.gamesnum)]
        # olddriver = res[0]
        # youdriver = res[1]
        # stubbon = res[2]
        # randomer = res[3]
        # explorer = res[4]
        # plt.figure(dpi = 200)
        # plt.plot(step, olddriver, 'r--', label='olddriver')
        # plt.plot(step, youdriver, 'g--', label='youdriver')
        # plt.plot(step, stubbon, 'b--', label='stubbon')
        # plt.plot(step, randomer, 'y--', label='randomer')
        # plt.plot(step, explorer, 'm--', label='explorer')
        # plt.title('Number of player VS. game process (old:35%, young: 35%)')
        # plt.xlabel('steps')
        # plt.ylabel('Number of player')
        # plt.legend()
        # plt.show()
        self.res = res
        return res
    
    def plot(self):
        step = [i for i in range(self.gamesnum)]
        olddriver = self.res[0]
        youdriver =self.res[1]
        stubbon = self.res[2]
        randomer = self.res[3]
        explorer = self.res[4]
        plt.figure(dpi = 200)
        plt.plot(step, olddriver, 'r--', label='olddriver')
        plt.plot(step, youdriver, 'g--', label='youdriver')
        plt.plot(step, stubbon, 'b--', label='stubbon')
        plt.plot(step, randomer, 'y--', label='randomer')
        plt.plot(step, explorer, 'm--', label='explorer')
        plt.title('Number of player VS. game process (old:35%, young: 35%)')
        plt.xlabel('steps')
        plt.ylabel('Number of player')
        plt.legend()
        plt.show()
        
#%%
a = Evolution(2500, 201, 1000, 500)
a.game()
a.plot()
#%%Traning level verification
len1 = len(range(500, 3000, 500))
olddiver1 = []

#200 is the number of game play
step = [i for i in range(1000)]
train_level = [i for i in range(500, 3000, 1500)]
seed(1234)
for i in range(500, 3000, 500):
    a = Evolution(i, 201, 1000, 500)
    res1 = a.game()
    olddiver1.append(res1[0])
l1 = plt.plot(step, olddiver1[0], 'r--', label='train level = 500')
l2 = plt.plot(step, olddiver1[1], 'g--', label='train level = 1000')
l3 = plt.plot(step, olddiver1[2], 'b--', label='train level = 1500')
l4 = plt.plot(step, olddiver1[3], 'y--', label='train level = 2000')
l5 = plt.plot(step, olddiver1[4], 'm--', label='train level = 2500')
plt.title('Number of olddiver over time VS. train level(error = 0.3)')
plt.xlabel('steps')
plt.ylabel('Number of olddiver')
plt.legend()
plt.show()