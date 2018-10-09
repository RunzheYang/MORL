import os
import sys
import argparse

""" DETAILS:
    # THIS FILE EXPLORES GP/REGR/FF/LSTM MODELS
        -- Try varying AT LEAST the following network parameters:
            a) network structures: n_hideen, L1, L2, acitivation 
            b) learning rate, decay, and regularisation
"""

################################################
### repository path
################################################
repository_path = os.path.abspath(os.path.join(os.getcwd(),'../../../'))


def config_text(domains, root, seed,
                screen_level,
                maxturns,
                belieftype, useconfreq, policytype, startwithhello, inpolicyfile, outpolicyfile, learning,
                maxiter, gamma, learning_rate, tau, replay_type, minibatch_size, capacity,
                exploration_type, epsilon_start, epsilon_end, n_in, features, max_k, \
                learning_algorithm, architecture, h1_size, h2_size,
                kernel,
                random, scale,
                usenewgoalscenarios,
                nbestsize,
                patience,
                penaliseallturns,
                wrongvenuepenalty,
                notmentionedvaluepenalty,
                sampledialogueprobs,
                save_step,
                confscorer,
                oldstylepatience,
                forcenullpositive,
                file_level,
                maxinformslots,
                informmask,
                informcountaccepted,
                requestmask, confusionmodel, byemask,
                n_samples, alpha_divergence, alpha, sigma_eps, sigma_prior,
                stddev_var_mu, stddev_var_logsigma, mean_log_sigma,
                nbestgeneratormodel,
                delta, beta, is_threshold, train_iters_per_episode, training_frequency):

    text = '[GENERAL]' + '\n'
    text += 'domains = ' + domains + '\n'
    text += 'singledomain = True' + '\n'
    text += 'root = ' + root + '\n'
    text += 'seed = ' + seed + '\n'
    text += '\n'

    text += '[conditional]' + '\n'
    text += 'conditionalsimuser = True\n'
    text += 'conditionalbeliefs = True\n'
    text += '\n'

    text += '[agent]' + '\n'
    text += 'maxturns = ' + maxturns + '\n'
    text += '\n'

    text += '[logging]' + '\n'
    text += 'screen_level = ' + screen_level + '\n'
    text += 'file_level = ' + file_level + '\n'
    text += '\n'

    text += '[simulate]' + '\n'
    text += 'mindomainsperdialog = 1\n'
    text += 'maxdomainsperdialog = 1\n'
    text += 'forcenullpositive = ' + forcenullpositive + '\n'
    text += '\n'

    text += '[policy]' + '\n'
    text += 'maxinformslots = ' + maxinformslots + '\n'
    text += 'informmask = ' + informmask + '\n'
    text += 'informcountaccepted = ' + informcountaccepted + '\n'
    text += 'requestmask = ' + requestmask + '\n'
    text += 'byemask = ' + byemask + '\n'
    text += '\n'

    text += '[policy_' + domains + ']' + '\n'
    text += 'belieftype = ' + belieftype + '\n'
    text += 'useconfreq = ' + useconfreq + '\n'
    text += 'policytype = ' + policytype + '\n'
    text += 'startwithhello = ' + startwithhello + '\n'
    text += 'inpolicyfile = ' + inpolicyfile + '\n'
    text += 'outpolicyfile = ' + outpolicyfile + '\n'
    text += 'learning = ' + learning + '\n'
    text += 'save_step = ' + save_step + '\n'
    text += '\n'

    text += '[dqnpolicy_' + domains + ']' + '\n'
    text += 'maxiter = ' + maxiter + '\n'
    text += 'gamma = ' + gamma + '\n'
    text += 'learning_rate = ' + learning_rate + '\n'
    text += 'tau = ' + tau + '\n'
    text += 'replay_type = ' + replay_type + '\n'
    text += 'minibatch_size = ' + minibatch_size + '\n'
    text += 'capacity = ' + capacity + '\n'
    text += 'exploration_type = ' + exploration_type + '\n'
    text += 'epsilon_start = ' + epsilon_start + '\n'
    text += 'epsilon_end = ' + epsilon_end + '\n'
    text += 'n_in = ' + n_in + '\n'
    text += 'features = ' + features + '\n'
    text += 'max_k = ' + max_k + '\n'
    text += 'learning_algorithm = ' + learning_algorithm + '\n'
    text += 'architecture = ' + architecture + '\n'
    text += 'h1_size = ' + h1_size + '\n'
    text += 'h2_size = ' + h2_size + '\n'
    text += 'training_frequency = ' + training_frequency + '\n'

    # BDQN
    text += 'n_samples = ' + n_samples + '\n'
    text += 'stddev_var_mu = ' + stddev_var_mu + '\n'
    text += 'stddev_var_logsigma = ' + stddev_var_logsigma + '\n'
    text += 'mean_log_sigma = ' + mean_log_sigma + '\n'
    text += 'sigma_prior = ' + sigma_prior + '\n'
    text += 'alpha =' + alpha + '\n'
    text += 'alpha_divergence =' + alpha_divergence + '\n'
    text += 'sigma_eps = ' + sigma_eps + '\n'

    # ACER
    text += 'delta = ' + delta + '\n'
    text += 'beta = ' + beta + '\n'
    text += 'is_threshold = ' + is_threshold + '\n'
    text += 'train_iters_per_episode = ' + train_iters_per_episode + '\n'
    text += '\n'

    text += '[gppolicy_' + domains + ']' + '\n'
    text += 'kernel = ' + kernel + '\n'
    text += '\n'

    text += '[gpsarsa_' + domains + ']' + '\n'
    text += 'random = ' + random + '\n'
    text += 'scale = ' + scale + '\n'
    text += '\n'

    text += '[usermodel]' + '\n'
    text += 'usenewgoalscenarios = ' + usenewgoalscenarios + '\n'
    text += 'sampledialogueprobs = ' + sampledialogueprobs + '\n'
    text += 'oldstylepatience = ' + oldstylepatience + '\n'
    text += '\n'

    text += '[errormodel]' + '\n'
    text += 'nbestsize = ' + nbestsize + '\n'
    text += 'confusionmodel = ' + confusionmodel + '\n'
    text += 'nbestgeneratormodel = ' + nbestgeneratormodel + '\n'
    text += 'confscorer = ' + confscorer + '\n'
    text += '\n'

    text += '[goalgenerator]' + '\n'
    text += 'patience = ' + patience + '\n'
    text += '\n'

    text += '[eval]' + '\n'
    text += 'rewardvenuerecommended = 0' + '\n'
    text += 'penaliseallturns = ' + penaliseallturns + '\n'
    text += 'wrongvenuepenalty = ' + wrongvenuepenalty + '\n'
    text += 'notmentionedvaluepenalty = ' + notmentionedvaluepenalty + '\n'
    text += '\n'

    text += '[eval_' + domains + ']' + '\n'
    text += 'successmeasure = objective' + '\n'
    text += 'successreward = 20' + '\n'
    text += '\n'

    return text


def run_on_grid(targetDir, step, iter_in_step, test_iter_in_step, parallel, execDir, configName, text, mode,
                error):

    ################################################
    ### config file
    config = repository_path + configName + '.cfg'

    # if directory not exist, then creat one
    config_dir = repository_path + 'configures/'
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)

    with open(config, 'w') as f:
        f.write(text)

    runStr = 'running ' + config
    print '{0:*^60}'.format(runStr)

    # command = 'python run_grid_pyGPtraining_rpg.py ' + targetDir + ' 3 10000 1 ' + execDir + ' 15 1 ' + config
    if mode == ('train', 'grid'):
        command = 'python run_grid_pyGPtraining_rpg.py ' + targetDir + ' ' + step + ' ' + \
                  iter_in_step + ' ' + parallel + ' ' + execDir + ' ' + error + ' 1 ' + config
    elif mode == ('test', 'grid'):
        command = 'python run_grid_pyGPtraining_rpg_test.py ' + targetDir + ' TEST ' + step + ' ' + \
                  test_iter_in_step + ' ' + parallel + ' ' + execDir + ' ' + error + ' 1 ' + config
    elif mode == ('train', 'own'):
        command = 'python run_own_pyGPtraining_rpg.py ' + targetDir + ' ' + step + ' ' + \
                  iter_in_step + ' ' + parallel + ' ' + execDir + ' ' + error + ' 1 ' + config
    elif mode == ('test', 'own'):
        command = 'python run_own_pyGPtraining_rpg_test.py ' + targetDir + ' TEST ' + step + ' ' + \
                  test_iter_in_step + ' ' + parallel + ' ' + execDir + ' ' + error + ' 1 ' + config

    print command

    os.system(command)

def main(argv):
    step = '20'
    iter_in_step = '200'
    test_iter_in_step = '200'
    save_step = '200'
    parallel = '1'
    maxiter = str(int(step) * int(iter_in_step))

    ################################################
    ###  Domain information
    ################################################
    domains = 'CamRestaurants'  # SF restaurants
    if len(argv) > 4:
        repository_path = argv[4]
    root = repository_path
    seed = argv[3]

    screen_level = 'warning'
    file_level = 'warning'
    maxturns = '25'

    ################################################
    ###  General policy information
    ################################################
    belieftype = 'focus'
    useconfreq = 'False'
    policytype_vary = ['tracer'] #'['dqn', 'a2c', 'enac', 'bdqn', 'acer']  # gp dqn bdqn acer
    startwithhello = 'False'
    inpolicyfile = 'policyFile'
    outpolicyfile = 'policyFile'
    learning = 'True'

    maxinformslots = '5'  # Maximum number of slot values that are presented in the inform summary action
    informmask = 'True'  # Decides if the mask over inform type actions is used or not (having the mask active speeds up learning)
    informcountaccepted = '4'  # number of accepted slots needed to unmask the inform_byconstraints action
    requestmask = 'True'  # Decides if the mask over inform type actions is used or not
    byemask = 'True'

    ################################################
    ###  DNN architecture options
    ################################################
    gamma = '0.99'  # discount factor
    learning_rate = '0.001'  # learning rate
    tau_vary = ['0.02']  # target policy network update frequency 0.02 is equal to update policy after 50 epochs
    replay_type_vary = ['vanilla']  # ['vanilla'] experience replay
    minibatch_size_vary = ['64']  # how many turns are in the batch
    capacity_vary = ['1000']  # how many turns/dialogues are in ER
    exploration_type_vary = ['e-greedy']  # 'e-greedy', 'Boltzman'
    epsilon_s_e_vary = [('0.9', '0.0')]  # , ('0.3', '0.0')]#, ('0.5', '0.1')]
    training_frequency = '2'  # how often train the model, episode_count % frequency == 0
    features = '["discourseAct", "method", "requested", "full", "lastActionInformNone", "offerHappened", "inform_info"]'
    max_k = '5'
    learning_algorithm = 'dqn'
    architecture = 'vanilla'
    h1_size = ['130']#, '200', '300']
    h2_size = ['50']#, '75', '100']

    ################################################
    ###  Bayesian estimation parameters
    ################################################
    n_samples = '1'  # number of samples for action choice
    alpha_divergence = 'False'  # use alpha divergence?
    alpha = '0.85'

    sigma_eps = '0.01'  # variance size for sampling epsilon
    sigma_prior = '1.5'  # prior for variance in KL term
    stddev_var_mu = '0.01'  # stdv for weights
    stddev_var_logsigma = '0.01'  # stdv of variance for variance
    mean_log_sigma = '0.000001'  # prior mean for variance

    ################################################
    ###  ACER parameters
    ################################################
    beta = '0.95'
    delta = '1.0'
    is_threshold = '5.0'
    train_iters_per_episode = '1'

    ################################################
    ###  User model and environment model info.
    ################################################
    usenewgoalscenarios = 'True'
    sampledialogueprobs = 'True'
    confusionmodel = 'RandomConfusions'
    confscorer = 'additive'  # 'additive'
    nbestgeneratormodel = 'SampledNBestGenerator'
    nbestsize = '1'
    patience = '3'
    penaliseallturns = 'True'
    wrongvenuepenalty = '0'
    notmentionedvaluepenalty = '0'

    oldstylepatience = 'True'
    forcenullpositive = 'False'
    runError_vary = ['0']
    # runError_vary = ['0' , '15', '30', '45', '50']

    if domains is 'CamRestaurants':
        n_in = '268'
    elif domains is 'CamHotels':
        n_in = '111'
    elif domains is 'SFRestaurants':
        n_in = '636'
    elif domains is 'SFHotels':
        n_in = '438'
    elif domains is 'Laptops11':
        n_in = '257'
    elif domains is 'TV':
        n_in = '188'
    elif domains is 'Booking':
        n_in = '188'

    ################################################
    ###  GP policy training options
    ################################################
    kernel = 'polysort'
    random = 'False'
    scale = '3'

    ConfigCounter = 0

    listFile = open(argv[0], 'w')
    runMode = ('train', 'grid')

    if argv[1] not in ('train', 'test') or argv[2] not in ('grid', 'own'):
        print '\n!!!!! WRONG COMMAND !!!!!\n'
        print 'EXAMPLE: python runScript.py list [train|test] [grid|own]\n'
        exit(1)
    elif argv[1] == 'train':
        if argv[2] == 'grid':
            runMode = ('train', 'grid')
        elif argv[2] == 'own':
            runMode = ('train', 'own')
    elif argv[1] == 'test':
        if argv[2] == 'grid':
            runMode = ('test', 'grid')
        elif argv[2] == 'own':
            runMode = ('test', 'own')

    listOutput = '{0: <6}'.format('PARAM') + '\t'
    listOutput += '{0: <10}'.format('type') + '\t'
    listOutput += '{0: <10}'.format('actor_lr') + '\t'
    listOutput += '{0: <10}'.format('critic_lr') + '\t'
    listOutput += '{0: <10}'.format('replaytype') + '\t'
    listOutput += '{0: <10}'.format('nMini') + '\t'
    listOutput += '{0: <10}'.format('capacity') + '\t'
    listOutput += '{0: <10}'.format('runError') + '\t'

    listFile.write(listOutput + '\n')

    for policytype in policytype_vary:
        for tau in tau_vary:
            for replay_type in replay_type_vary:
                for minibatch_size in minibatch_size_vary:
                    for exploration_type in exploration_type_vary:
                        for capacity in capacity_vary:
                            for epsilon_s_e in epsilon_s_e_vary:
                                epsilon_start, epsilon_end = epsilon_s_e
                                for h1 in h1_size:
                                    for h2 in h2_size:
                                        for runError in runError_vary:

                                            execDir = repository_path

                                            if policytype == 'gp':
                                                targetDir = 'CamRestaurants_gp_'
                                            elif policytype == 'dqn' or policytype == 'dqn_vanilla':
                                                targetDir = 'CamRestaurants_dqn_'
                                            elif policytype == 'a2c':
                                                targetDir = 'CamRestaurants_a2c_'
                                            elif policytype == 'enac':
                                                targetDir = 'CamRestaurants_enac_'
                                            elif policytype == 'bdqn':
                                                targetDir = 'CamRestaurants_bdqn_'
                                            elif policytype == 'acer':
                                                targetDir = 'CamRestaurants_acer_'
                                            elif policytype == 'a2cis':
                                                targetDir = 'CamRestaurants_a2cis_'
                                            elif policytype == 'tracer':
                                                targetDir = 'CamRestaurants_tracer_'

                                            listOutput = '{0: <10}'.format(targetDir) + '\t'
                                            listOutput += '{0: <10}'.format(policytype) + '\t'
                                            listOutput += '{0: <10}'.format(learning_rate) + '\t'
                                            listOutput += '{0: <10}'.format(replay_type) + '\t'
                                            listOutput += '{0: <10}'.format(minibatch_size) + '\t'
                                            listOutput += '{0: <10}'.format(capacity) + '\t'
                                            listOutput += '{0: <10}'.format(runError) + '\t'

                                            targetDir += 'learning_rate' + learning_rate + '_replay_type' + replay_type + \
                                                         '_minibatch_size' + minibatch_size + '_capacity' + capacity + '_runError' + runError

                                            text = config_text(domains, root, seed,
                                                    screen_level,
                                                    maxturns,
                                                    belieftype, useconfreq, policytype, startwithhello,
                                                    inpolicyfile, outpolicyfile, learning,
                                                    maxiter, gamma, learning_rate, tau, replay_type,
                                                    minibatch_size, capacity,
                                                    exploration_type, epsilon_start, epsilon_end, n_in,
                                                    features, max_k, learning_algorithm, architecture, h1,
                                                    h2,
                                                    kernel,
                                                    random, scale,
                                                    usenewgoalscenarios,
                                                    nbestsize,
                                                    patience,
                                                    penaliseallturns,
                                                    wrongvenuepenalty,
                                                    notmentionedvaluepenalty,
                                                    sampledialogueprobs,
                                                    save_step,
                                                    confscorer,
                                                    oldstylepatience,
                                                    forcenullpositive,
                                                    file_level,
                                                    maxinformslots, informmask,informcountaccepted,requestmask, confusionmodel, byemask,
                                                    n_samples, alpha_divergence, alpha, sigma_eps, sigma_prior,
                                                    stddev_var_mu, stddev_var_logsigma, mean_log_sigma,
                                                    nbestgeneratormodel,
                                                    delta, beta, is_threshold, train_iters_per_episode, training_frequency)

                                            # run_on_grid(targetDir, execDir, configName, text)
                                            tmpName = 'gRun' + str(ConfigCounter)

                                            run_on_grid(tmpName, step, iter_in_step, test_iter_in_step, parallel, execDir, tmpName, text,
                                                        runMode, runError)

                                            listFile.write(tmpName + '\t' + listOutput + '\n')
                                            ConfigCounter += 1


if __name__ == "__main__":
    argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description='DeepRL parameter search')
    parser.add_argument('-s', '--seed', help='set the random seed', required=False, type=str, default="123")
    parser.add_argument('-tn', '--train', help='script is set to train policies (default)', action='store_true')
    parser.add_argument('-tt', '--test', help='script is set to test/evaluate policies', action='store_true')

    parser.add_argument('--own', help='run on local machine (default)', action='store_true')
    parser.add_argument('--grid', help='run on grid', action='store_true')

    parser.add_argument('-f', '--file', help='the list file', required=False, type=str, default='list')

    parser.add_argument('-p', '--pydial', help='the path to pydial', required=False, type=str, default='../../../')


    if len(argv) > 0 and not argv[0][0] == '-':
        if len(sys.argv) != 5:
            parser.print_help()
            # print '\n!!!!! WRONG COMMAND !!!!!\n'
            # print 'EXAMPLE: python runScript.py list [train|test] [grid|own]\n'
            exit(1)
        # main(argv)
    else:
        # parser = argparse.ArgumentParser(description='DeepRL parameter search')
        # parser.add_argument('-s', '--seed', help='set the random seed', required=False, type=str, default="123")
        # parser.add_argument('-tn', '--train', help='script is set to train policies (default)', action='store_true')
        # parser.add_argument('-tt', '--test', help='script is set to test/evaluate policies', action='store_true')

        # parser.add_argument('--own', help='run on local machine (default)', action='store_true')
        # parser.add_argument('--grid', help='run on grid', action='store_true')

        # parser.add_argument('-f', '--file', help='the list file', required=False, type=str, default='list')

        # parser.add_argument('-p', '--pydial', help='the path to pydial', required=False, type=str, default='../../../')

        args = parser.parse_args()
        
        own = not args.grid
        grid = not args.own and args.grid
        if own == grid:
            pass # issue error with parameter help
        
        train = not args.test
        test = not args.train and args.test
        if train == test:
            pass # issue error with parameter help

        pydialpath = os.path.abspath(os.path.join(os.getcwd(),args.pydial))

        argv = [args.file, 'test' if test else 'train', 'grid' if grid else 'own', args.seed, pydialpath]

    # print argv

    main(argv)
# END OF FILE 

