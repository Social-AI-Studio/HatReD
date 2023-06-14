import argparse 

def parse_train_opt(parser):
    parser.add_argument('--SAVE_FILEPATH', type=str, required=True)

def parse_test_opt(parser):
    parser.add_argument('--SAVE_FILEPATH', type=str, required=True)
    parser.add_argument('--ENC_MODEL_FILEPATH', type=str)#,default='roberta-large')
    parser.add_argument('--DEC_MODEL_FILEPATH', type=str)#,default='roberta-large')

def parse_opt(phase):
    parser=argparse.ArgumentParser()
    parser.add_argument('--DATASET',type=str,default='fhm')
    
    #path configuration
    parser.add_argument('--TRAIN_ANNOTATIONS',type=str, required=True)
    parser.add_argument('--TEST_ANNOTATIONS',type=str, required=True)
    parser.add_argument('--CAPTIONS_FILEPATH',type=str)
    parser.add_argument('--FEATURES_FILEPATH',type=str)
    parser.add_argument('--RESULT',type=str,default='./result')
    
    #hyper parameters configuration
    parser.add_argument('--WEIGHT_DECAY',type=float,default=0.01) 
    parser.add_argument('--LR_RATE',type=float,default=1e-5) 
    parser.add_argument('--EPS',type=float,default=1e-8) 
    parser.add_argument('--BATCH_SIZE',type=int,default=16)
    parser.add_argument('--FIX_LAYERS',type=int,default=0)

    parser.add_argument('--USE_ENTITIES', action="store_true")
    parser.add_argument('--USE_DEMOGRAPHICS', action="store_true")
    parser.add_argument('--USE_UNDERSTANDINGS', action="store_true")
    parser.add_argument('--USE_CAPTIONS', action="store_true")
    parser.add_argument('--USE_TITLE', action="store_true")
    parser.add_argument('--USE_SNIPPET', action="store_true")
    parser.add_argument('--USE_SNIPPET_EXTENDED', action="store_true")
    parser.add_argument('--CLEAN_CONTRACTIONS', action="store_true")
    parser.add_argument('--TIE_WEIGHTS', action="store_true")
    parser.add_argument('--TRAIN', action="store_true")
    
    parser.add_argument('--DEBUG', action="store_true")
    parser.add_argument('--SAVE',type=bool,default=False)
    parser.add_argument('--EPOCHS',type=int,default=15)
    
    parser.add_argument('--SEED', type=int, default=1111, help='random seed')
    
    # Encoder-Decoder Model Settings
    parser.add_argument('--WARM_UP',type=int,default=0)

    # Model Settings
    parser.add_argument('--PRETRAIN_MODEL_NAME', type=str)#,default='roberta-large')
    parser.add_argument('--ENC_MODEL_NAME', type=str)#,default='roberta-large')
    parser.add_argument('--DEC_MODEL_NAME', type=str)#,default='roberta-large')
    parser.add_argument('--GRADIENT_ACCUMULATION_STEPS',type=int,default=1)
    parser.add_argument('--CUDA_DEVICE', type=str, required=True)

    # Dataset Settings
    parser.add_argument('--TARGET_MODE', type=str, required=True) # 'first', 'random'
    parser.add_argument('--TEST_MODEL_FILEPATH', type=str)#,default='roberta-large')

    if phase == "train":
        parse_train_opt(parser)
    
    if phase == "test":
        parse_test_opt(parser)
    
    args=parser.parse_args()
    return args
