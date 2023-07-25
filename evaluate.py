import os
import json
import numpy,random
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from scipy.special import comb, perm
from sklearn import preprocessing
def MDS_visual(path_name):
    path = os.path.dirname(__file__)
    # concat21-triplet-loss_result.json
    data_path=os.path.join(path,"results",path_name)
    with open(data_path) as f:
        data=json.load(f)


    # 构建数据列表
    # 3471:3481
    data_list=list(data.values())
    data_arr=numpy.array(data_list)

    # 获取数据集编号、数据故事编号
    data_text=list()
    data_dataset=list()
    for i in list(data.keys()):
        re=i.split("-",1)
        data_dataset.append(re[0])
        data_text.append(re[1])
    
    # 根据数据集，构建颜色列表
    duplicate_dataset=list(set(data_dataset))
    color_dict={}
    color_label=list()
    for dataset in duplicate_dataset:
        color_dict[dataset]=randomcolor()
    for item in data_dataset:
        color_label.append(color_dict[item])


    # 采用TSNE/MDS降维到二维平面
    # data_embedded = TSNE(n_components=2,learning_rate='auto', init='pca', random_state=0).fit_transform(data_arr)
    data_embedded = MDS(n_components=2,  random_state=0).fit_transform(data_arr)
    x_min, x_max = data_embedded.min(0), data_embedded.max(0)
    data_embedded = (data_embedded - x_min) / (x_max - x_min)

    x=data_embedded[:,0]
    y=data_embedded[:,1]
    for i in range(len(x)):
        plt.text(x[i],y[i],data_text[i],size=7)
    plt.scatter(x = data_embedded[:,0],y = data_embedded[:,1], c=color_label)
    plt.show()



def randomcolor():
    colorArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0,14)]
    return "#"+color

def evaluate_nearest_dis_triplets(path_name):
    path = os.path.dirname(__file__)
    data_path=os.path.join(path,"results",path_name)
    with open(data_path) as f:
        data=json.load(f)
    
    # data=remove_len_lower2_dataset(data)
    same_dataset_facts={}
    accuracy_list_window1=list()
    accuracy_list_window2=list()
    accuracy_list_window3=list()
    accuracy_list_story=list()

    for key in data.keys():
        dataset_label=key.split("-",1)[0]
        if len(same_dataset_facts)==0 or (len(same_dataset_facts)>0 and dataset_label==list(same_dataset_facts.keys())[0].split("-",1)[0]):
            same_dataset_facts[key]=data[key]
        else:
            # 1.先计算上一个dataset中facts之间的准确率
            accuracy1,accuracy2,accuracy3,accuracy_s=cal_same_dataset_facts_dis(same_dataset_facts)
            if (accuracy1>1) or(accuracy2>1) or (accuracy3>1) or (accuracy_s>1) :
                print("error!")
            accuracy_list_window1.append(accuracy1)
            accuracy_list_window2.append(accuracy2)
            accuracy_list_window3.append(accuracy3)
            accuracy_list_story.append(accuracy_s)
            # 2. 设置为空，添加新的
            same_dataset_facts={}
            same_dataset_facts[key]=data[key]
    # 3. 计算最后一个dataset的准确率
    accuracy1,accuracy2,accuracy3,accuracy_s=cal_same_dataset_facts_dis(same_dataset_facts)
    accuracy_list_window1.append(accuracy1)
    accuracy_list_window2.append(accuracy2)
    accuracy_list_window3.append(accuracy3)
    accuracy_list_story.append(accuracy_s)

    all_dataset_mean_accuracy1=numpy.mean(accuracy_list_window1)
    all_dataset_mean_accuracy2=numpy.mean(accuracy_list_window2)
    all_dataset_mean_accuracy3=numpy.mean(accuracy_list_window3)
    all_dataset_mean_accuracy_s=numpy.mean(accuracy_list_story)

    all_dataset_mean_max1=numpy.max(accuracy_list_window1)
    all_dataset_mean_max2=numpy.max(accuracy_list_window2)

    # 检测最高数据比例的源自哪个数据集
    # dataset_id_list=[]
    # for key in data.keys():
    #     dataset_label=key.split("-",1)[0]
    #     if  dataset_label not in dataset_id_list:
    #         dataset_id_list.append(dataset_label)
    # max_index=numpy.argmax(accuracy_list_window1)
    # max_acc_dataset=dataset_id_list[max_index]
    # print(max_acc_dataset)
    print(all_dataset_mean_accuracy1,  all_dataset_mean_accuracy2, all_dataset_mean_accuracy3,all_dataset_mean_accuracy_s)
    return all_dataset_mean_accuracy1,  all_dataset_mean_accuracy2, all_dataset_mean_accuracy3,all_dataset_mean_accuracy_s

def cal_same_dataset_facts_dis(facts_dict):
    dis_map={}
    facts_dict_key=list(facts_dict.keys())
    for i in range(len(facts_dict)):
        key1 = facts_dict_key[i]
        fact1_id=key1.split("-",1)[1]
        value1=numpy.array(facts_dict[key1])
        min_dis=1e7
        record_min_factid=0
        for j in range(len(facts_dict)):
            if i==j: continue
            key2 = facts_dict_key[j]
            fact2_id=key2.split("-",1)[1]
            value2=numpy.array(facts_dict[key2])
            dis = eucliDist(value1,value2)
            if dis<min_dis:
                min_dis=dis
                record_min_factid=fact2_id
            
        dis_map[fact1_id+"#"+record_min_factid]=min_dis

    # 计算两两是否是上下文：
    is_context=0
    is_context_window2=0
    is_context_window3=0
    is_context_story=0
    for key in dis_map.keys():
        fact1_id=key.split("#")[0]
        fact2_id=key.split("#")[1]
        if fact1_id.split("-")[0] != fact2_id.split("-")[0]:
            continue
        if int(fact1_id.split("-")[1])+1==int(fact2_id.split("-")[1]) or int(fact1_id.split("-")[1])-1==int(fact2_id.split("-")[1]):
            is_context+=1
            is_context_window2+=1
            is_context_window3+=1
        if int(fact1_id.split("-")[1])+2==int(fact2_id.split("-")[1]) or int(fact1_id.split("-")[1])-2==int(fact2_id.split("-")[1]):
            is_context_window2+=1
            is_context_window3+=1
        if int(fact1_id.split("-")[1])+3==int(fact2_id.split("-")[1]) or int(fact1_id.split("-")[1])-3==int(fact2_id.split("-")[1]):
            is_context_window3+=1
        if int(fact1_id.split("-")[0])==int(fact2_id.split("-")[0]):
            is_context_story+=1

    window1_accuracy=float(is_context/len(dis_map))
    window2_accuracy=float(is_context_window2/len(dis_map))
    window3_accuracy=float(is_context_window3/len(dis_map))
    story_accuracy=float(is_context_story/len(dis_map))

    return window1_accuracy,window2_accuracy,window3_accuracy,story_accuracy


# 搜索最近的多个并存储
def evaluate_more_nearest_dis_triplets(path_name,search_num):
    path = os.path.dirname(__file__)
    data_path=os.path.join(path,path_name)
    with open(data_path) as f:
        data=json.load(f)
    
    same_dataset_facts={}
    accuracy_list_window1={}
    accuracy_list_window2={}
    accuracy_list_window3={}
    accuracy_list_story={}
    last_dataset_key=""
    for key in list(data.keys()):
        dataset_label=key.split("-",1)[0]
        if len(same_dataset_facts)==0 or (len(same_dataset_facts)>0 and dataset_label==list(same_dataset_facts.keys())[0].split("-",1)[0]):
            same_dataset_facts[key]=data[key]
            last_dataset_key=key.split("-",1)[0]
        else:
            # 1.先计算上一个dataset中facts之间的准确率
            accuracy1,accuracy2,accuracy3,accuracy_s=cal_same_dataset_facts_more_dis_min(same_dataset_facts,search_num)
            if (accuracy1>1) or(accuracy2>1) or (accuracy3>1) or (accuracy_s>1) :
                print("error!")
            accuracy_list_window1[last_dataset_key]=accuracy1
            accuracy_list_window2[last_dataset_key]=accuracy2
            accuracy_list_window3[last_dataset_key]=accuracy3
            accuracy_list_story[last_dataset_key]=accuracy_s
            # 2. 设置为空，添加新的
            same_dataset_facts={}
            same_dataset_facts[key]=data[key]
    # 3. 计算最后一个dataset的准确率
    accuracy1,accuracy2,accuracy3,accuracy_s=cal_same_dataset_facts_more_dis_min(same_dataset_facts,search_num)
    accuracy_list_window1[key]=accuracy1
    accuracy_list_window2[key]=accuracy2
    accuracy_list_window3[key]=accuracy3
    accuracy_list_story[key]=accuracy_s

    accuracy_list_window1=dict(sorted(accuracy_list_window1.items(), key=lambda d: d[1],reverse=True) )
    accuracy_list_window2=dict(sorted(accuracy_list_window2.items(), key=lambda d: d[1],reverse=True) )
    accuracy_list_window3=dict(sorted(accuracy_list_window3.items(), key=lambda d: d[1],reverse=True) )
    accuracy_list_story=dict(sorted(accuracy_list_story.items(), key=lambda d: d[1],reverse=True) )

    all_dataset_mean_accuracy1=numpy.mean(list(accuracy_list_window1.values()))
    all_dataset_mean_accuracy2=numpy.mean(list(accuracy_list_window2.values()))
    all_dataset_mean_accuracy3=numpy.mean(list(accuracy_list_window3.values()))
    all_dataset_mean_accuracy_s=numpy.mean(list(accuracy_list_story.values()))

    print("context window=1:",all_dataset_mean_accuracy1)
    print("context window=2:",all_dataset_mean_accuracy2)
    print("context window=3:",all_dataset_mean_accuracy3)
    print("context in the same story:",all_dataset_mean_accuracy_s)
    
    return all_dataset_mean_accuracy1,  all_dataset_mean_accuracy2, all_dataset_mean_accuracy3,all_dataset_mean_accuracy_s

def cal_same_dataset_facts_more_dis_min(facts_dict,search_num):
    dis_map={}
    facts_dict_key=list(facts_dict.keys())
    for i in range(len(facts_dict)):
        temp_dis_map={}
        key1 = facts_dict_key[i]
        fact1_id=key1.split("-",1)[1]
        value1=numpy.array(facts_dict[key1])
        for j in range(len(facts_dict)):
            if i==j: continue
            key2 = facts_dict_key[j]
            fact2_id=key2.split("-",1)[1]
            value2=numpy.array(facts_dict[key2])
            dis = eucliDist(value1,value2)
            if len(temp_dis_map.keys())<search_num:
                temp_dis_map[fact2_id]=dis
            else:
                # 如果已经存储了值，找到最大的那个，如果比它小，则替换掉。
                temp_max_key=max(temp_dis_map,key=temp_dis_map.get)
                if dis<temp_dis_map[temp_max_key] and dis>0:
                    del temp_dis_map[temp_max_key]
                    temp_dis_map[fact2_id]=dis
            
            
        dis_map[fact1_id]=temp_dis_map


    # 计算两两是否是上下文：
    is_context=0
    is_context_window2=0
    is_context_window3=0
    is_context_story=0
    for key in dis_map.keys():
        fact1_id=key
        for fact2 in dis_map[key].keys():
            fact2_id=fact2
            if fact1_id.split("-")[0] != fact2_id.split("-")[0]:
                continue
            
            if int(fact1_id.split("-")[1])+1==int(fact2_id.split("-")[1]) or int(fact1_id.split("-")[1])-1==int(fact2_id.split("-")[1]):
                is_context+=1
                is_context_window2+=1
                is_context_window3+=1
                is_context_story+=1
                break
            if int(fact1_id.split("-")[1])+2==int(fact2_id.split("-")[1]) or int(fact1_id.split("-")[1])-2==int(fact2_id.split("-")[1]):
                is_context_window2+=1
                is_context_window3+=1
                is_context_story+=1
                break
            if int(fact1_id.split("-")[1])+3==int(fact2_id.split("-")[1]) or int(fact1_id.split("-")[1])-3==int(fact2_id.split("-")[1]):
                is_context_window3+=1
                is_context_story+=1
                break
            if int(fact1_id.split("-")[0])==int(fact2_id.split("-")[0]):
                is_context_story+=1
                break
            

    window1_accuracy=float(is_context/len(dis_map.keys()))
    window2_accuracy=float(is_context_window2/len(dis_map.keys()))
    window3_accuracy=float(is_context_window3/len(dis_map.keys()))
    story_accuracy=float(is_context_story/len(dis_map.keys()))
    a=story_accuracy

    return window1_accuracy,window2_accuracy,window3_accuracy,story_accuracy


def eucliDist(A,B):
    scalar=preprocessing.MinMaxScaler()
    # A=numpy.array(A).reshape(-1,1)
    # B=numpy.array(B).reshape(-1,1)
    # normA=scalar.fit_transform(A)
    # normB=scalar.fit_transform(B)
    return numpy.sqrt(sum(numpy.power((A - B), 2)))


def cal_random_p():
    path = os.path.dirname(__file__)
    data_path=os.path.join(path,"dataset/new_dataset","test_facts_labels.json")
    with open(data_path) as f:
        test_data=json.load(f)
    
    combined_test_data=combine_by_same_dataset(test_data)
    all_dataset_p=[]
    for key in combined_test_data.keys():
        dataset_len=0
        all_p=0
        for item in combined_test_data[key]:
            dataset_len+=len(combined_test_data[key][item])
        
        for item in combined_test_data[key]:
            story_p=cal_one_story_p(dataset_len,len(combined_test_data[key][item]),pick_num=1)
            all_p+=story_p

        p=all_p/dataset_len
        all_dataset_p.append(p)
        
    print(numpy.mean(all_dataset_p))

def remove_len_lower2_dataset(test_data):
    combined_data=combine_result_dataset(test_data)
    keeped_id=[]
    for dataset in combined_data.keys():
        if len(combined_data[dataset]) > 2:
             for story in combined_data[dataset].keys():
                 dataset_id=dataset
                 for fact in combined_data[dataset][story]:   
                    keeped_id.append(dataset_id+"-"+fact)
    
    print(keeped_id)
    new_test_data={}
    for item in keeped_id:
        new_test_data[item]=test_data[item]
        
    return new_test_data

def combine_result_dataset(data):
    new_data={}
    last_dataset_id="none"
    for fact_key in data.keys():
        dataset_id=fact_key.split("-")[0]
        story_id=fact_key.split("-")[1]
        fact_id=fact_key.split("-",1)[1]
        if dataset_id != last_dataset_id:
            last_dataset_id=dataset_id
            new_data[last_dataset_id]={}
        if story_id not in new_data[last_dataset_id]:
            new_data[last_dataset_id][story_id]=[]
        new_data[last_dataset_id][story_id].append(fact_id)
    return new_data

def combine_by_same_dataset(data):
    new_data={}
    last_dataset_id="none"
    for fact in data:
        fact_key=fact["fact_id"]
        dataset_id=fact_key.split("-")[0]
        story_id=fact_key.split("-")[1]
        fact_id=fact_key.split("-",1)[1]
        if dataset_id != last_dataset_id:
            last_dataset_id=dataset_id
            new_data[last_dataset_id]={}
        if story_id not in new_data[last_dataset_id]:
            new_data[last_dataset_id][story_id]=[]
        new_data[last_dataset_id][story_id].append(fact_id)
    return new_data

def cal_one_story_p(dataset_story_len,current_story_len,pick_num):
    one_fact=1-float(comb(dataset_story_len-current_story_len,pick_num)/comb(dataset_story_len-1,pick_num))
    return one_fact*current_story_len

def cal_one_story_p_window1(dataset_story_len,current_story_len,pick_num):
    complete_set=dataset_story_len-1
    end_point=dataset_story_len-2
    center_point=dataset_story_len-3
    head_or_tail=1-float(comb(end_point,pick_num)/comb(complete_set,pick_num))
    center=1-float(comb(center_point,pick_num)/comb(complete_set,pick_num))
    return head_or_tail*2+center*(current_story_len-2)

def cal_one_story_p_window2(dataset_story_len,current_story_len,pick_num):
    complete_set=dataset_story_len-1
    end_point=dataset_story_len-3
    end_center_point=dataset_story_len-4
    center_point=dataset_story_len-5
    head_or_tail=1-float(comb(end_point,pick_num)/comb(complete_set,pick_num))
    near_center=1-float(comb(end_center_point,pick_num)/comb(complete_set,pick_num))
    center=1-float(comb(center_point,pick_num)/comb(complete_set,pick_num))
    return head_or_tail*2+near_center*2+center*(current_story_len-4)

def cal_one_story_p_window3(dataset_story_len,current_story_len,pick_num):

    num_list=[]
    if current_story_len==5:
        num_list=[1,2,3,2,1]
    elif current_story_len==6:
        num_list=[1,2,3,3,2,1]
    elif current_story_len==7:
        num_list=[1,2,3,4,3,2,1]
    elif current_story_len==8:
        num_list=[1,2,3,4,4,3,2,1]
    else:
        print("error!")
    complete_set=dataset_story_len-1
    other_story_len=dataset_story_len-current_story_len
    point1=numpy.maximum(current_story_len-4,0)+other_story_len
    point2=numpy.maximum(current_story_len-5,0)+other_story_len
    point3=numpy.maximum(current_story_len-6,0)+other_story_len
    point4=numpy.maximum(current_story_len-7,0)+other_story_len
    all_p=0
    for num in num_list:
        if num==1:
            all_p+=1-float(comb(point1,pick_num)/comb(complete_set,pick_num))
        elif num==2:
            all_p+=1-float(comb(point2,pick_num)/comb(complete_set,pick_num))
        elif num==3:
            all_p+=1-float(comb(point3,pick_num)/comb(complete_set,pick_num))
        elif num==4:
            all_p+=1-float(comb(point4,pick_num)/comb(complete_set,pick_num))
    return all_p

def factorial(num):
	return eval('*'.join(map(str,range(1,num+1))))

if __name__ =="__main__":
    # path_name="word2vec-concat12-avgpool-1_result.json"
    path_name="results/chart2vec_base-result_256.json"
    # path_name="./evaluation/quantitive-experiment/chartseer/write_json.json"
    # path_name="./evaluation/quantitive-experiment/erato-fact2vec/erato_result.json"
    evaluate_more_nearest_dis_triplets(path_name,search_num=1)
    # evaluate_nearest_dis_triplets(path_name)
    # cal_random_p()
    # MDS_visual(path_name)