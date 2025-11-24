from Main_MTGNN import run_pipeline
import time

if __name__ == "__main__":
    config_name = "Korea_train.yaml"
    db_name = 'Mongo_db'
    course = [1, 1, 1] 
    # 1 = skip, 1st = data load from DB / 2nd = data preprocess / 3rd = map gen 
    
    run_pipeline(mode="test",version='v1.0.0',config_name=config_name,course=course,db_name=db_name,
                 cluster_ids =None)#[0,1,2,3,4,5,6,7,8,9,10])
