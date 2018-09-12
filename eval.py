import os
from collections import defaultdict
import pickle

"""从标签序列中整理出字典，字典的键是实体名称，键值是list,
其中每个元素是实体的开始位置和终止位置的坐标对"""

nameDic = {"B-ORG":"I-ORG","B-LOC":"I-LOC","B-PER":"I-PER"}

def get_evaDic(seq,nameDic):

    evalDic = defaultdict(list)
    for n,s in enumerate(seq):
        if s in nameDic.keys():
            endName = nameDic[s]
            endCoor = n
            while True:
                endCoor += 1
                if seq[endCoor] != endName:
                    endCoor -= 1
                    break   
            evalDic[s].append((n,endCoor))

    return evalDic


def conlleval(label_predict, label_path, metric_path):
    """

    :param label_predict:
    :param label_path:
    :param metric_path:
    :return:
    """
    true_tags = []
    predict_tags = []

    eval_perl = "./conlleval_rev.pl"
    with open(label_path, "w") as fw:
        line = []
        for sent_result in label_predict:
            for char, tag, tag_ in sent_result:
                tag = '0' if tag == 'O' else tag
                char = char.encode("utf-8")
                true_tags.append(tag)
                predict_tags.append(tag_)
                line.append("{} {} {}\n".format(char, tag, tag_))
            true_tags.append('end')
            predict_tags.append('end')
            line.append("\n")
        fw.writelines(line)

    trueDic = get_evaDic(true_tags,nameDic)
    predictDic = get_evaDic(predict_tags,nameDic)

    with open('trueDic.pkl','wb') as f:
        pickle.dump(trueDic,f)
    with open('predictDic','wb') as f:
        pickle.dump(predictDic,f)

    precisionDic = {}
    recallDic = {}
    fDic = {}

    for key in nameDic.keys():
        precisionDic[key] = 0
        trueSet = set(trueDic[key])
        predictSet = set(predictDic[key])
        precision = len(trueSet & predictSet)/len(predictSet)
        recall = len(trueSet & predictSet)/len(trueSet)
        f = 2*precision*recall/(precision+recall)
        precisionDic[key],recallDic[key],fDic[key] = (precision,recall,f)

    with open(metric_path,'w') as f:
        for key in nameDic.keys():
            f.write(str(key)+"准确率为：%f"%(precisionDic[key])+'\n')
            f.write(str(key)+"召回率为：%f"%(precisionDic[key])+'\n')
            f.write(str(key)+"F1值为：%f"%(fDic[key])+'\n')
            f.write('\n')
    return precisionDic,recallDic,fDic


    """
    这里是之前作者的评估方式，实体识别的评估标准有很多，参见
    http://www.davidsbatista.net/blog/2018/05/09/Named_Entity_Evaluation/
    
    os.system("perl {} < {} > {}".format(eval_perl, label_path, metric_path))
    with open(metric_path) as fr:
        metrics = [line.strip() for line in fr]
    return metrics

    """
    