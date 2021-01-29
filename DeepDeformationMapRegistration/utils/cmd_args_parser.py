import sys, getopt
import DeepDeformationMapRegistration.utils.constants as C
import os


def parse_arguments(argv):

    try:
        opts, args = getopt.getopt(argv, "hg:b:l:r:d:t:i:f:x:p:q:", ["gpu-num=",
                                                             "batch-size=",
                                                             "loss=",
                                                             "remote=",
                                                             "debug=",
                                                             "debug-training=",
                                                             "debug-input-data=",
                                                             "destination-folder=",
                                                             "destination-folder-fix=",
                                                             "training-dataset=",
                                                             "test-dataset=",
                                                             "help"])
    except getopt.GetoptError:
        print('\n\t\t--gpu-num:\t\tGPU number to use'
              '\n\t\t--batch-size:\t\tsize of the training batch'
              '\n\t\t--loss:\t\tLoss function: ncc, mse, dssim'
              '\n\t\t--remote:\t\tExecuting the script in The Beast: "True"/"False". Def: False'
              '\n\t\t--debug:\t\tEnable debugging logs: "True"/"False". Def: False'
              '\n\t\t--debug-training:\t\tEnable debugging training logs: "True"/"False". Def: False'
              '\n\t\t--debug-input-data:\t\tEnable debugging input data logs: "True"/"False". Def: False'
              '\n\t\t--destination-folder:\t\tName of the folder where to save the generated training files'
              '\n\t\t--destination-folder-fixed:\t\tSame as --destination-folder but do not add the timestamp'
              '\n\t\t--training-dataset:\t\tPath to the training dataset file'
              '\n\t\t--test-dataset:\t\tPath to the test dataset file'
              '\n')
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('--help', '-h'):
            print('\n\t\t--gpu-num:\t\tGPU number to use\n\t\t--batch-size:\t\tsize of the training batch'
                  '\n\t\t--loss:\t\tLoss function: ncc, mse, dssim\n')
            continue
        elif opt in ('--gpu_num', '-g'):
            old = C.GPU_NUM
            C.GPU_NUM = arg
            os.environ['CUDA_VISIBLE_DEVICES'] = C.GPU_NUM
            print('\t\tGPU_NUM: {} -> {}'.format(old, C.GPU_NUM))

        elif opt in ('--batch-size', '-b'):
            old = C.BATCH_SIZE
            C.BATCH_SIZE = int(arg)
            print('\t\tBATCH_SIZE: {} -> {}'.format(old, C.BATCH_SIZE))

        elif opt in ('--destination-folder', '-f'):
            old = C.DESTINATION_FOLDER
            C.DESTINATION_FOLDER = arg + '_' + C.CUR_DATETIME
            print('\t\tDESTINATION_FOLDER: {} -> {}'.format(old, C.DESTINATION_FOLDER))

        elif opt in ('--destination-folder-fixed', '-x'):
            old = C.DESTINATION_FOLDER
            C.DESTINATION_FOLDER = arg
            print('\t\tDESTINATION_FOLDER: {} -> {}'.format(old, C.DESTINATION_FOLDER))

        elif opt in ('--training-dataset', '-p'):
            old = C.TRAINING_DATASET
            C.TRAINING_DATASET = arg
            print('\t\tTRAINING_DATASET: {} -> {}'.format(old, C.TRAINING_DATASET))

        elif opt in ('--test-dataset', '-q'):
            old = C.TEST_DATASET
            C.TEST_DATASET = arg
            print('\t\tTEST_DATASET: {} -> {}'.format(old, C.TEST_DATASET))

        elif opt in ('--remote', '-r'):
            old = C.REMOTE
            if arg.lower() in ('1', 'true', 't'):
                C.REMOTE = True
            else:
                C.REMOTE = False
            print('\t\tREMOTE: {} -> {}'.format(old, C.REMOTE))

        elif opt in ('--debug', '-d'):
            old = C.DEBUG
            if arg.lower() in ('1', 'true', 't'):
                C.DEBUG = True
            else:
                C.DEBUG = False
            print('\t\tDEBUG: {} -> {}'.format(old, C.DEBUG))

        elif opt in ('--debug-training', '-t'):
            old = C.DEBUG_TRAINING
            if arg.lower() in ('1', 'true', 't'):
                C.DEBUG_TRAINING = True
            else:
                C.DEBUG_TRAINING = False
            print('\t\tDEBUG_TRAINING: {} -> {}'.format(old, C.DEBUG_TRAINING))

        elif opt in ('--debug-input-data', '-i'):
            old = C.DEBUG_INPUT_DATA
            if arg.lower() in ('1', 'true', 't'):
                C.DEBUG_INPUT_DATA = True
            else:
                C.DEBUG_INPUT_DATA = False
            print('\t\tDEBUG_INPUT_DATA: {} -> {}'.format(old, C.DEBUG_INPUT_DATA))

        elif opt in ('--loss', '-l'):
            old = C.LOSS_FNC
            if arg in ('ncc', 'mse', 'dssim', 'dice'):
                C.LOSS_FNC = arg
            else:
                print('Invalid option for --loss. Expected: "mse", "ncc" or "dssim", got {}'.format(arg))
                sys.exit(2)
            print('\t\tLOSS_FNC: {} -> {}'.format(old, C.LOSS_FNC))
