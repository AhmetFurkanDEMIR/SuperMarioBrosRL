# @author: Ahmet Furkan DEMIR

def getEnv():

    # env name
    a = ["SuperMarioBros-v0", "SuperMarioBros-v1", "SuperMarioBros-v2", "SuperMarioBros-v3", "SuperMarioBros2-v0", "SuperMarioBros2-v1"]

    return a[0] 

def getConfig():

    config = {
        
        'env':'super_mario_bros',
        'num_workers': 4, 
        'train_batch_size': 256,
        'num_gpus': 1,
        'framework': 'tf',
        'gamma': 0.99,
        'sigma0': 0.5,
        'dueling': True,
        'double_q': True,

        'model': {
            
            'custom_model': 'TFModel',
            'custom_model_config': {}, 
        },


    }

    return config
