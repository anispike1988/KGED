import scipy
import scipy.io
import random
import collections

from batching import *


def read_from_id(filename='/user_home/cwj1988/research/convkb_description/data/BIOFULL/cv10/2/entity2id.txt'):
    entity2id = {}
    id2entity = {}
    with open(filename) as f:
        for line in f:
            if len(line.strip().split()) > 1:
                tmp = line.strip().split()
                entity2id[tmp[0]] = int(tmp[1])
                id2entity[int(tmp[1])] = tmp[0]
    return entity2id, id2entity


def init_norm_Vector(relinit, entinit, embedding_size):
    lstent = []
    lstrel = []
    with open(relinit) as f:
        for line in f:
            tmp = [float(val) for val in line.strip().split()]
            # if np.linalg.norm(tmp) > 1:
            #     tmp = tmp / np.linalg.norm(tmp)
            lstrel.append(tmp)
    with open(entinit) as f:
        for line in f:
            tmp = [float(val) for val in line.strip().split()]
            # if np.linalg.norm(tmp) > 1:
            #     tmp = tmp / np.linalg.norm(tmp)
            lstent.append(tmp)

    #print ("len(lstent[0]) : " + str(len(lstent[0])))
    #print ("embedding_size : " + str(embedding_size ))
    assert embedding_size % len(lstent[0]) == 0
    return np.array(lstent, dtype=np.float32), np.array(lstrel, dtype=np.float32)

def init_norm_Vector_des(relinit, entinit, embedding_size):
    lstent = []
    lstrel = []
    with open(relinit) as f:
        for line in f:
            tmp = [float(val) for val in line.strip().split()]
            # if np.linalg.norm(tmp) > 1:
            #     tmp = tmp / np.linalg.norm(tmp)
            lstrel.append(tmp)
    with open(entinit) as f:
        for line in f:
            tmp = [float(val) for val in line.strip().split()]
            # if np.linalg.norm(tmp) > 1:
            #     tmp = tmp / np.linalg.norm(tmp)
            lstent.append(tmp)

    #print ("len(lstent[0]) : " + str(len(lstent[0])))
    #print ("embedding_size : " + str(embedding_size ))
    assert embedding_size % len(lstent[0]) == 0
    return np.array(lstent, dtype=np.float32), np.array(lstrel, dtype=np.float32)


def getID(folder='/user_home/cwj1988/research/convkb_description/data/BIOFULL/cv10/2/'):
    lstEnts = {}
    lstRels = {}
    with open(folder + 'train.txt') as f:
        for line in f:
            line = line.strip().split()
            if line[0] not in lstEnts:
                lstEnts[line[0]] = len(lstEnts)
            if line[2] not in lstEnts:
                lstEnts[line[2]] = len(lstEnts)
            if line[1] not in lstRels:
                lstRels[line[1]] = len(lstRels)

    with open(folder + 'valid.txt') as f:
        for line in f:
            line = line.strip().split()
            if line[0] not in lstEnts:
                lstEnts[line[0]] = len(lstEnts)
            if line[2] not in lstEnts:
                lstEnts[line[2]] = len(lstEnts)
            if line[1] not in lstRels:
                lstRels[line[1]] = len(lstRels)

    with open(folder + 'test.txt') as f:
        for line in f:
            line = line.strip().split()
            if line[0] not in lstEnts:
                lstEnts[line[0]] = len(lstEnts)
            if line[2] not in lstEnts:
                lstEnts[line[2]] = len(lstEnts)
            if line[1] not in lstRels:
                lstRels[line[1]] = len(lstRels)

    wri = open(folder + 'entity2id.txt', 'w')
    for entity in lstEnts:
        wri.write(entity + '\t' + str(lstEnts[entity]))
        wri.write('\n')
    wri.close()

    wri = open(folder + 'relation2id.txt', 'w')
    for entity in lstRels:
        wri.write(entity + '\t' + str(lstRels[entity]))
        wri.write('\n')
    wri.close()


def parse_line(line):
    line = line.strip().split()
    sub = line[0]
    rel = line[1]
    obj = line[2]
    val = [1]
    if len(line) > 3:
        if line[3] == '-1':
            val = [-1]
    return sub, obj, rel, val


def load_triples_from_txt(filename, words_indexes=None, parse_line=parse_line):
    """
    Take a list of file names and build the corresponding dictionnary of triples
    """
    if words_indexes == None:
        words_indexes = dict() #key_value 
        #print("word_indexes = " , words_indexes) = {}
        entities = set() #no redundant = {}
        #print("entities = " , entities)
        next_ent = 0
    else:
        entities = set(words_indexes)
        #print("entities = " , entities) #{'06671637', '02784998'. '08014615', .......,;00588780'}
        next_ent = max(words_indexes.values()) + 1
        #print("next_ent = " , next_ent) # 40768

    data = dict()
    #data = collections.OrderedDict()

    with open(filename) as f:
        lines = f.readlines() #= all triplet data in xxxx.txt 
        
    for _, line in enumerate(lines):
        sub, obj, rel, val = parse_line(line) #sub = head , obj = tail, rel = relation, val = 1 (true triplet) in each line
        #print(sub +"\t" + rel +"\t" + obj)
        if sub in entities:
            sub_ind = words_indexes[sub]
            #print("sub_ind : ", sub_ind) #index in entity2id.txt ex>12844
            #print("sub : ", sub) #entity id ex>06308049
        else: # store entities that are not in eneities variable.
            sub_ind = next_ent
            next_ent += 1
            words_indexes[sub] = sub_ind
            entities.add(sub)

        if rel in entities:
            rel_ind = words_indexes[rel]
        else:
            rel_ind = next_ent
            next_ent += 1
            words_indexes[rel] = rel_ind
            entities.add(rel)

        if obj in entities:
            obj_ind = words_indexes[obj]
        else:
            obj_ind = next_ent
            next_ent += 1
            words_indexes[obj] = obj_ind
            entities.add(obj)

        data[(sub_ind, rel_ind, obj_ind)] = val # give 1 to true triplet

    indexes_words = {} #key : 18616, value : 05219724
    for tmpkey in words_indexes: 
        indexes_words[words_indexes[tmpkey]] = tmpkey
        #print("words_indexes[tmpkey] : ", words_indexes[tmpkey]) #index in entity2id.txt
        #print("tmpkey : ", tmpkey) #entity id ex>06308049

    return data, words_indexes, indexes_words


def build_data(name='BIO20FULL', path='../data'):
    folder = path + '/' + name + '/'
    
    #MAKE DATA SUITABLE FOR THE MODEL USING load_triples_from_txt FUNCTION.
    train_triples, words_indexes, _ = load_triples_from_txt(folder + 'train.txt', parse_line=parse_line)
    #print("train_triples : " + str(train_triples))


    valid_triples, words_indexes, _ = load_triples_from_txt(folder + 'valid.txt',
                                                            words_indexes=words_indexes, parse_line=parse_line)

    test_triples, words_indexes, indexes_words = load_triples_from_txt(folder + 'test.txt',
                                                                       words_indexes=words_indexes,
                                                                       parse_line=parse_line)

    entity2id, id2entity = read_from_id(folder + '/entity2id.txt')
    relation2id, id2relation = read_from_id(folder + '/relation2id.txt')
    left_entity = {}
    right_entity = {}

    with open(folder + 'train.txt') as f:
        lines = f.readlines()
    for _, line in enumerate(lines):
        head, tail, rel, val = parse_line(line)
        # count the number of occurrences for each (heal, rel)
        if relation2id[rel] not in left_entity:
            left_entity[relation2id[rel]] = {}
        if entity2id[head] not in left_entity[relation2id[rel]]:
            left_entity[relation2id[rel]][entity2id[head]] = 0
        left_entity[relation2id[rel]][entity2id[head]] += 1
        # count the number of occurrences for each (rel, tail)
        if relation2id[rel] not in right_entity:
            right_entity[relation2id[rel]] = {}
        if entity2id[tail] not in right_entity[relation2id[rel]]:
            right_entity[relation2id[rel]][entity2id[tail]] = 0
        right_entity[relation2id[rel]][entity2id[tail]] += 1

    left_avg = {}
    for i in range(len(relation2id)):
        left_avg[i] = sum(left_entity[i].values()) * 1.0 / len(left_entity[i])

    right_avg = {}
    for i in range(len(relation2id)):
        right_avg[i] = sum(right_entity[i].values()) * 1.0 / len(right_entity[i])

    headTailSelector = {}
    for i in range(len(relation2id)):
        headTailSelector[i] = 1000 * right_avg[i] / (right_avg[i] + left_avg[i])

    #print("words_indexes : " + str(words_indexes))

    # wordidx = open("/user_home/cwj1988/research/convkb_description/data/BIOFULL/cv10/2/wordes_indexs.txt", "w") #added
    # wordidx.write(str(words_indexes)) #added   

    # wordidx.close()

    return train_triples, valid_triples, test_triples, words_indexes, indexes_words, headTailSelector, entity2id, id2entity, relation2id, id2relation

def dic_of_chars(words_indexes):
    lstChars = {}
    for word in words_indexes:
        for char in word:
            if char not in lstChars:
                lstChars[char] = len(lstChars)
    lstChars['unk'] = len(lstChars)
    return lstChars


def convert_to_seq_chars(x_batch, lstChars, indexes_words):
    lst = []
    for [tmpH, tmpR, tmpT] in x_batch:
        wH = [lstChars[tmp] for tmp in indexes_words[tmpH]]
        wR = [lstChars[tmp] for tmp in indexes_words[tmpR]]
        wT = [lstChars[tmp] for tmp in indexes_words[tmpT]]
        lst.append([wH, wR, wT])
    return lst

def _pad_sequences(sequences, pad_tok, max_length):
    sequence_padded, sequence_length = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
        sequence_padded += [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


def pad_sequences(sequences, pad_tok):
    sequence_padded, sequence_length = [], []
    max_length_word = max([max(map(lambda x: len(x), seq))
                           for seq in sequences])
    for seq in sequences:
        # all words are same length now
        sp, sl = _pad_sequences(seq, pad_tok, max_length_word)
        sequence_padded += [sp]
        sequence_length += [sl]

    max_length_sentence = max(map(lambda x: len(x), sequences))
    sequence_padded, _ = _pad_sequences(sequence_padded, [pad_tok] * max_length_word, max_length_sentence)
    sequence_length, _ = _pad_sequences(sequence_length, 0, max_length_sentence)

    return np.array(sequence_padded).astype(np.int32), np.array(sequence_length).astype(np.int32)

