import copy
import numpy as np
import random as rnd
import matplotlib.pyplot as plt

class table():
    def __init__(self,agent_1 = None,agent_2 = None):
        self.agent_1 = agent_1
        self.agent_2 = agent_2
        self.constraint_table = np.random.randint(low =1 , high = 11, size=(10,10))
class message():
    def __init__(self,sender,receiver,senderValue):
        self.sender = sender
        self.receiver = receiver
        self.senderValue = senderValue
class Agent():
    def __init__(self,ID):
        self.ID = ID
        self.list_constraint_tables = []
        self.my_value = rnd.randint(0,9)
        self.cost = 290
        self.inbox = {}
        self.temp_gain = None
        self.temp_value = None
        self.neighbors_gain = np.zeros(shape=30,dtype=int)


    def calc_cost(self):
        costs_vector = np.zeros(shape=10,dtype=int)
        # Extracting table with neighbor
        for i in self.inbox:
            table_with_neighb = self.get_table(i)
            neighbor_val = self.inbox[i]
            costs_vector = np.add(costs_vector,table_with_neighb[:,neighbor_val])
        best_value = np.argmin(costs_vector)
        return best_value , (self.cost - costs_vector[best_value])
    def get_table(self, key):
        for i in self.list_constraint_tables:
            if self.ID == i.agent_1 and key == i.agent_2:
                return i.constraint_table



def init_problems(p1,p2,amount_of_problems):
    problems_set_1 = []
    problems_set_2 = []
    for i in range(10):
        temp_agents = init_agents()
        for j in temp_agents:
            init_constraint_connections(temp_agents, p1, j)
        problems_set_1.append(temp_agents)
    for i in range(10):
        temp_agents = init_agents()
        for j in temp_agents:
            init_constraint_connections(temp_agents, p2, j)
        problems_set_2.append(temp_agents)
    return problems_set_1,problems_set_2
def init_agents():
    list_of_agents = []
    for i in range(30):
        list_of_agents.append(Agent(i))
    return list_of_agents
def init_constraint_connections(list_agents,prob ,currAgent):
    for i in range(currAgent.ID + 1,30):
        coin_flip = rnd.random()
        if coin_flip <= prob:
           const_table = table(currAgent.ID,i)
           currAgent.list_constraint_tables.append(const_table)

           transpose_table = table(i,currAgent.ID)
           shallow_copy = copy.copy(const_table.constraint_table)
           transpose_table.constraint_table = np.transpose(shallow_copy)
           list_agents[i].list_constraint_tables.append(transpose_table)
def send_to_agents(mailbox,list_of_agents):
    for i in mailbox:
        list_of_agents[i.receiver].inbox[i.sender]=i.senderValue
def send_to_mailbox(list_of_agents):
    mailbox = []
    for k in list_of_agents:
        for j in k.list_constraint_tables:
            msg = message(k.ID,j.agent_2,k.my_value)
            mailbox.append(msg)
    return mailbox


def MGM_algorithm(list_of_agents,amount_of_problems):
    iteration_cost = np.ndarray(100,dtype=int)
    total_cost = np.ndarray(100,dtype=float)
    for s in range(amount_of_problems):
        agents = list_of_agents[s]
        for i in range(100):
            mail = send_to_mailbox(agents)
            send_to_agents(mail,agents)
            for j in agents:
                new_value,gain = j.calc_cost()
                j.temp_gain = gain
                j.temp_value = new_value
                for k in j.inbox:
                    agents[k].neighbors_gain[j.ID] = j.temp_gain
            for j in agents:
                if j.temp_gain >= np.amax(j.neighbors_gain):
                    j.my_value = j.temp_value
                    j.cost = j.cost - j.temp_gain
            totalCost = 0
            for p in agents:
                for k in p.list_constraint_tables:
                    table_with_neighbor = k.constraint_table
                    neighbor = agents[k.agent_2]
                    totalCost += (table_with_neighbor[p.my_value][neighbor.my_value]) / 2
            iteration_cost[i] = totalCost
        total_cost = np.add(total_cost, iteration_cost)
    total_cost = np.divide(total_cost, np.full(100, 10))
    return total_cost
def DSA_algorithm(list_of_problems , improvement_prob,amount_of_problems):
    iteration_cost = np.ndarray(100,dtype=int)
    total_cost = np.ndarray(100,dtype=float)
    for s in range(amount_of_problems):
        agents = list_of_problems[s]
        for i in range(100):
            mail = send_to_mailbox(agents)
            send_to_agents(mail,agents)
            for j in agents:
                new_value,gain = j.calc_cost()
                if gain > 0 and rnd.random() < improvement_prob:
                        j.my_value = new_value
                        j.cost = j.cost - gain
            totalCost = 0
            for p in agents:
                for k in p.list_constraint_tables:
                    table_with_neighbor = k.constraint_table
                    neighbor = agents[k.agent_2]
                    totalCost += (table_with_neighbor[p.my_value][neighbor.my_value])/2
            iteration_cost[i] = totalCost
        total_cost = np.add(total_cost,iteration_cost)
    total_cost = np.divide(total_cost,np.full(100,10))
    return total_cost



problems_1,problems_2 = init_problems(p1 = 0.2 , p2 = 0.5 , amount_of_problems= 20)
prb_1 = copy.deepcopy(problems_1)
prb_2 = copy.deepcopy(problems_1)
prb_3 = copy.deepcopy(problems_1)
prb_4 = copy.deepcopy(problems_2)
prb_5 = copy.deepcopy(problems_2)
prb_6 = copy.deepcopy(problems_2)


iterations = np.arange(start=0, stop=100, step=1)

result_1 = DSA_algorithm(prb_1,0.4,10)
plt.plot(result_1,label ="DSA p1=0.2 p2=0.4" ,color="red")
result_2 = DSA_algorithm(prb_2,0.7,10)
plt.plot(result_2, label ="DSA p1=0.2 p2=0.7" ,color="yellow")
result_3 = MGM_algorithm(prb_3,10)
plt.plot(result_3,label ="MGM p1=0.2", color="blue")
plt.legend()
plt.show()

result_4 = DSA_algorithm(prb_4,0.4,10)
plt.plot(result_4,label ="DSA p1=0.5 p2=0.4" ,color="red")
result_5 = DSA_algorithm(prb_5,0.7,10)
plt.plot(result_5, label ="DSA p1=0.5 p2=0.7" ,color="yellow")
result_6 = MGM_algorithm(prb_6,10)
plt.plot(result_6,label ="MGM p1=0.5", color="blue")
plt.legend()
plt.show()



