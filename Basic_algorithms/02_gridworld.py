states = [i for i in range(16)]
values = [0 for _ in range(16)]
actions = ['n', 'w', 's', 'e']
ds_actions = {"n": -4, "e":1, "s":4, "w":-1}
gamma = 1.

def nextState(s, a):
    next_state = s
    if (s % 4 == 0 and a == 'w') or (s < 4 and a =='n') or \
        ((s+1) %4 == 0 and a == 'e') or (s > 11 and a == 's'):
        pass
    else:
        ds = ds_actions[a]
        next_state = s + ds

    return next_state

def rewardOf(s):
    return 0 if s in [0, 15] else -1

def isTerminateState(s):
    return s in [0, 15]  

def getSuccessors(s):
    successors = []
    if isTerminateState(s):
        return successors
    for a in actions:
        next_state = nextState(s, a)
        successors.append(next_state)
    return successors

# update the value of state s
def updateValue(s):
    sucessors = getSuccessors(s)
    newValue = 0  # values[s]
    reward = rewardOf(s)
    for next_state in sucessors:
        newValue += 1./len(sucessors) * (reward + gamma * values[next_state])
    return newValue

def performOneIteration():
    newValue = [0 for _ in range(16)]
    for s in states:
        newValue[s] = updateValue(s)
    global values
    values = newValue
    printValue(values)

# show some array info of the small grid world
def printValue(v):
    for i in range(16):
        print('{0:>6.2f}'.format(v[i]),end = " ")
        if (i+1)%4 == 0:
            print("\n")
    print()

# test function
def test():
    printValue(states)
    printValue(values)
    for s in states:
        reward = rewardOf(s)
        for a in actions:
            next_state = nextState(s, a)
            print("({0}, {1}) -> {2}, with reward {3}".format(s, a,next_state, reward))

    for i in range(200):
        performOneIteration()
        printValue(values)

def main():
    max_iterate_times = 160
    cur_iterate_times = 0
    while cur_iterate_times <= max_iterate_times:
        print("Iterate No.{0}".format(cur_iterate_times))
        performOneIteration()
        cur_iterate_times += 1
    printValue(values)

if __name__ == '__main__':
    main()




