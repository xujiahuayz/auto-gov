num_pools = 3
num_action_pool = 3

for action in range(num_action_pool**num_pools):
    actions = []
    for i in [0, 1, 2]:
        exponent = num_pools - i
        this_action = (
            action
            % (num_action_pool**exponent)
            // (num_action_pool ** (exponent - 1))
        )
        actions.append(this_action)

    action_reverse = 0
    for i, this_action in enumerate(actions):
        action_reverse += this_action * num_action_pool ** (num_pools - i - 1)

    print(action_reverse == action)
