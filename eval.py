import tensorflow as tf
import numpy as np
np.random.seed(1234)
import os
import time
import datetime
from builddata import *
from model import ConvKB
# Parameters
# ==================================================
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

tf.flags.DEFINE_string("data", "DATA_FOLDER_PATH", "Data sources.")
tf.flags.DEFINE_string("run_folder", "RUN_FOLDER_NAME", "Data sources.")
tf.flags.DEFINE_string("name", "DATA_FOLDER_NAME", "Name of the dataset.")

tf.flags.DEFINE_integer("embedding_dim", 50, "Dimensionality of character embedding")
tf.flags.DEFINE_integer("embedding_des_dim", 512, "Dimensionality of character embedding")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes")
tf.flags.DEFINE_integer("num_filters", 500, "Number of filters per filter size")
tf.flags.DEFINE_float("dropout_keep_prob", 0.7, "Dropout keep probability")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0001, "L2 regularization lambda")
tf.flags.DEFINE_float("learning_rate", 0.0001, "Learning rate")
tf.flags.DEFINE_boolean("is_trainable", True, "")
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size")
tf.flags.DEFINE_float("neg_ratio", 1.0, "Number of negative triples generated by positive")
tf.flags.DEFINE_boolean("use_pretrained", True, "Using the pretrained embeddings")
tf.flags.DEFINE_integer("num_epochs", 201, "Number of training epochs")
tf.flags.DEFINE_integer("saveStep", 200, "")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_string("model_name", 'MODEL_NAME', "")
tf.flags.DEFINE_boolean("useConstantInit", False, "")

tf.flags.DEFINE_string("model_index", '200', "")
tf.flags.DEFINE_integer("num_splits", 2, "Split the validation set into 8 parts for a faster evaluation")
tf.flags.DEFINE_integer("testIdx", 0, "From 0 to 7. Index of one of 8 parts")
tf.flags.DEFINE_boolean("decode", True, "")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# Load data
print("Loading data...")

train, valid, test, words_indexes, indexes_words, \
    headTailSelector, entity2id, id2entity, relation2id, id2relation = build_data(path=FLAGS.data, name=FLAGS.name)
data_size = len(train)
train_batch = Batch_Loader(train, words_indexes, indexes_words, headTailSelector, \
                           entity2id, id2entity, relation2id, id2relation, batch_size=FLAGS.batch_size, neg_ratio=FLAGS.neg_ratio)


print("train_batch : " + str(train_batch))

entity_array = np.array(list(train_batch.indexes_ents.keys()))  # [0   2    3   ... , 40952, 40952, 40953]


lstEmbed = []
lstEmbed_des = []
if FLAGS.use_pretrained == True:
    print("Using pre-trained model.")
    lstEmbed = np.empty([len(words_indexes), FLAGS.embedding_dim]).astype(np.float32)
    lstEmbed_des = np.empty([len(words_indexes), FLAGS.embedding_des_dim]).astype(np.float32)
    initEnt, initRel = init_norm_Vector(FLAGS.data + FLAGS.name + '/relation2vec' + str(FLAGS.embedding_dim) + '.init',
                                            FLAGS.data + FLAGS.name + '/entity2vec' + str(FLAGS.embedding_dim) + '.init', FLAGS.embedding_dim)
    initEnt_des, initRel_des = init_norm_Vector_des(FLAGS.data + FLAGS.name + '/relationdescription2vec' + str(FLAGS.embedding_des_dim) + '.init', FLAGS.data + FLAGS.name + '/entitydescription2vec' + str(FLAGS.embedding_des_dim) + '.init', FLAGS.embedding_des_dim)                                            
    for _word in words_indexes:
        if _word in relation2id:
            index = relation2id[_word]
            _ind = words_indexes[_word]
            lstEmbed[_ind] = initRel[index]
        elif _word in entity2id:
            index = entity2id[_word]
            _ind = words_indexes[_word]
            lstEmbed[_ind] = initEnt[index]
        else:
            print('*****************Error********************!')
            break

    for _word in words_indexes:
        if _word in relation2id:
            index = relation2id[_word]
            _ind = words_indexes[_word]
            lstEmbed_des[_ind] = initRel_des[index]
        elif _word in entity2id:
            index = entity2id[_word]
            _ind = words_indexes[_word]
            lstEmbed_des[_ind] = initEnt_des[index]
        else:
            print('*****************Error********************!')
            break
    lstEmbed = np.array(lstEmbed, dtype=np.float32)
    lstEmbed_des = np.array(lstEmbed_des, dtype=np.float32)

print("Loading data... finished!")

x_valid = np.array(list(valid.keys())).astype(np.int32)
y_valid = np.array(list(valid.values())).astype(np.float32)
len_valid = len(x_valid)
batch_valid = int(len_valid / (FLAGS.num_splits - 1))

x_test = np.array(list(test.keys())).astype(np.int32)
y_test = np.array(list(test.values())).astype(np.float32)
len_test = len(x_test)
batch_test = int(len_test / (FLAGS.num_splits - 1)) # FLAGS.num_splits = 8
print("batch_test : " , batch_test) # = 178 (1250/8-1)

##########################################

if FLAGS.decode == False:
    lstModelNames = list(FLAGS.model_name.split(","))
    for _model_name in lstModelNames:
        out_dir = os.path.abspath(os.path.join(FLAGS.run_folder, "runs", _model_name))
        print("Evaluating {}\n".format(out_dir))
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")

        lstModelIndexes = list(FLAGS.model_index.split(","))
        for _model_index in lstModelIndexes:
            _file = checkpoint_prefix + "-" + _model_index
            lstHT = []
            for _index in range(FLAGS.num_splits):
                with open(_file + '.eval.' + str(_index) + '.txt') as f:
                    for _line in f:
                        if _line.strip() != '':
                            lstHT.append(list(map(float, _line.strip().split())))
            lstHT = np.array(lstHT)
            print(_file, 'mr, mrr, hits@10 --> ',  np.sum(lstHT, axis=0)/(2 * len_test))

        print('------------------------------------')

else:
    with tf.Graph().as_default():
        tf.set_random_seed(1234)
        session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                      log_device_placement=FLAGS.log_device_placement)
        #session_conf.gpu_options.allow_growth = True
        session_conf.gpu_options.per_process_gpu_memory_fraction = 0.3
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            global_step = tf.Variable(0, name="global_step", trainable=False)



            #CNN START
            cnn = ConvKB(
                sequence_length=x_valid.shape[1], #3
                num_classes=y_valid.shape[1], #1
                pre_trained=lstEmbed,
                pre_trained_des=lstEmbed_des,
                embedding_size=FLAGS.embedding_dim,
                embedding_des_size=FLAGS.embedding_des_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                vocab_size=len(words_indexes),
                l2_reg_lambda=FLAGS.l2_reg_lambda,
                batch_size=(int(FLAGS.neg_ratio) + 1)*FLAGS.batch_size,
                is_trainable=FLAGS.is_trainable,
                useConstantInit=FLAGS.useConstantInit)
                

            # Output directory for models and summaries

            lstModelNames = list(FLAGS.model_name.split(","))

            for _model_name in lstModelNames:

                out_dir = os.path.abspath(os.path.join(FLAGS.run_folder, "runs", _model_name))
                print("Evaluating {}\n".format(out_dir))

                # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
                checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
                checkpoint_prefix = os.path.join(checkpoint_dir, "model")

                lstModelIndexes = list(FLAGS.model_index.split(","))

                for _model_index in lstModelIndexes:

                    _file = checkpoint_prefix + "-" + _model_index

                    cnn.saver.restore(sess, _file)

                    print("Loaded model", _file)

                    # Predict function to predict scores for test data


                    def predict(x_batch, y_batch, writer=None):
                        feed_dict = {
                            cnn.input_x: x_batch,
                            cnn.input_y: y_batch,
                            cnn.dropout_keep_prob: 1.0,
                        }
                        #scores = sess.run([cnn.predictions], feed_dict)
                        scores = sess.run([cnn.scores], feed_dict)
                        return scores

                    def test_prediction_left(x_batch, y_batch, head_or_tail='head'):

                        hits10 = 0.
                        mrr = 0.0
                        mr = 0.0
                        f_errl_rank = open("/KGED/data/test_errl_prediction_result_idx"+str(FLAGS.testIdx)+".txt", "a") #added
                        f_errl = open("/KGED/data/test_errl_prediction_result_idx_all"+str(FLAGS.testIdx)+".txt", "a") #added

                        for i in range(len(x_batch)):
                            mr_temp = 0.0
                            new_x_batch = np.tile(x_batch[i], (len(entity2id), 1)) #np.tile : repeat matries as 40943 by 1
                            f_errl_rank.write(str(x_batch[i])+"\t")
                            f_errl.write(str(x_batch[i])+"\t")
                            print("x_batch : " +str(x_batch[i]))
                            #print(len(entity2id)) #=40943
                            #print("new_x_batch : ", new_x_batch, "\n") 
                            #[21923   1  22358]
                            #[21923   1  22358]
                            #...
                            #[21923   1  22358]
                            #==> 40943
                            new_y_batch = np.tile(y_batch[i], (len(entity2id), 1)) 
                            #print ("new_y_batch : ", new_y_batch, "\n")
                            #[1.]
                            #[1.]
                            #..
                            #[1.]
                            if head_or_tail == 'head':
                                new_x_batch[:, 0] = entity_array
                            else:  # 'tail'
                                new_x_batch[:, 2] = entity_array
                            
                            while len(new_x_batch) % ((int(FLAGS.neg_ratio) + 1) * FLAGS.batch_size) != 0:
                                new_x_batch = np.append(new_x_batch, [x_batch[i]], axis=0)
                                new_y_batch = np.append(new_y_batch, [y_batch[i]], axis=0)

                            if head_or_tail == 'head':
                                entity_array1 = new_x_batch[:, 0]
                            else:  # 'tail'
                                entity_array1 = new_x_batch[:, 2]

                            results = []
                            listIndexes = range(0, len(new_x_batch), (int(FLAGS.neg_ratio) + 1) * FLAGS.batch_size)
                            for tmpIndex in range(len(listIndexes) - 1):
                                results = np.append(results, predict(
                                    new_x_batch[listIndexes[tmpIndex]:listIndexes[tmpIndex + 1]],
                                    new_y_batch[listIndexes[tmpIndex]:listIndexes[tmpIndex + 1]]))
                            results = np.append(results,
                                                predict(new_x_batch[listIndexes[-1]:], new_y_batch[listIndexes[-1]:]))

                            results = np.reshape(results, [entity_array1.shape[0], 1])
                            #print("results : " + str(results)) # scroe

                            results_with_id = np.hstack(
                                (np.reshape(entity_array1, [entity_array1.shape[0], 1]), results))
                            results_with_id = results_with_id[np.argsort(results_with_id[:, 1])]
                            results_with_id = results_with_id[:, 0].astype(int)
                            #print ("results_with_id : ", results_with_id, "\n")
                            #=[38498 20357 7786 ... .... 27602 1049]
                            #print ("len(results_with_id) : ", len(results_with_id)) 
                            #= 40960 fixed




                            _filter = 0
                            if head_or_tail == 'head':
                                for tmpHead in results_with_id:
                                    if tmpHead == x_batch[i][0]:
                                        f_errl.write(str(tmpHead) +",")
                                        break
                                    tmpTriple = (tmpHead, x_batch[i][1], x_batch[i][2])
                                    if (tmpTriple in train) or (tmpTriple in valid) or (tmpTriple in test):
                                        continue
                                    else:
                                        _filter += 1
                                        f_errl.write(str(tmpHead) +",")
                            else:
                                for tmpTail in results_with_id:
                                    if tmpTail == x_batch[i][2]:
                                        break
                                    tmpTriple = (x_batch[i][0], x_batch[i][1], tmpTail)
                                    if (tmpTriple in train) or (tmpTriple in valid) or (tmpTriple in test):
                                        continue
                                    else:
                                        _filter += 1

                            mr += (_filter + 1)
                            mr_temp += (_filter + 1)
                            #print ("mr : ", mr, "\n")
                            #= 3185.0
                            f_errl_rank.write(str(mr_temp)+"\n") #added
                            f_errl.write("\n") #added


                            mrr += 1.0 / (_filter + 1)
                            if _filter < 10:
                                hits10 += 1

                        f_errl.close()
                        f_errl_rank.close()

                        return np.array([mr, mrr, hits10])


                    def test_prediction_right(x_batch, y_batch, head_or_tail='tail'):

                        hits10 = 0.
                        mrr = 0.0
                        mr = 0.0
                        f_errr_rank = open("/KGED/dat/test_errr_prediction_result_idx"+str(FLAGS.testIdx)+".txt", "a") #added
                        f_errr = open("/KGED/data/test_errr_prediction_result_idx_all"+str(FLAGS.testIdx)+".txt", "a") #added
                        for i in range(len(x_batch)):
                            begin = time.clock()
                            mr_temp = 0.0
                            new_x_batch = np.tile(x_batch[i], (len(entity2id), 1)) #np.tile : repeat matries as 40943 by 1
                            f_errr.write(str(x_batch[i])+"\t")
                            f_errr_rank.write(str(x_batch[i])+"\t")
                            print("x_batch : " +str(x_batch[i]) +"         (" +str(i)+" out of " +str(len(x_batch))+")")
                            #print(len(entity2id)) #=40943
                            #print("new_x_batch : ", new_x_batch, "\n") 
                            #[21923   1  22358]
                            #[21923   1  22358]
                            #...
                            #[21923   1  22358]
                            #==> 40943
                            new_y_batch = np.tile(y_batch[i], (len(entity2id), 1)) 
                            #print ("new_y_batch : ", new_y_batch, "\n")
                            #[1.]
                            #[1.]
                            #..
                            #[1.]
                            if head_or_tail == 'head':
                                new_x_batch[:, 0] = entity_array
                            else:  # 'tail'
                                new_x_batch[:, 2] = entity_array
                            
                            while len(new_x_batch) % ((int(FLAGS.neg_ratio) + 1) * FLAGS.batch_size) != 0:
                                new_x_batch = np.append(new_x_batch, [x_batch[i]], axis=0)
                                new_y_batch = np.append(new_y_batch, [y_batch[i]], axis=0)

                            if head_or_tail == 'head':
                                entity_array1 = new_x_batch[:, 0]
                            else:  # 'tail'
                                entity_array1 = new_x_batch[:, 2]

                            results = []
                            listIndexes = range(0, len(new_x_batch), (int(FLAGS.neg_ratio) + 1) * FLAGS.batch_size)
                            for x in new_x_batch[:, 2]:
                                f_errr_with_score.write(str(x)+",")
                            f_errr_with_score.write("\t")


                            for tmpIndex in range(len(listIndexes) - 1):
                                results = np.append(results, predict(
                                    new_x_batch[listIndexes[tmpIndex]:listIndexes[tmpIndex + 1]],
                                    new_y_batch[listIndexes[tmpIndex]:listIndexes[tmpIndex + 1]]))
                            results = np.append(results,
                                                predict(new_x_batch[listIndexes[-1]:], new_y_batch[listIndexes[-1]:]))

                            results = np.reshape(results, [entity_array1.shape[0], 1])
                            #print("results : " + str(results)) # scroe

                            for sc in results:
                                f_errr_with_score.write(str(sc)+",")



                           
                            results_with_id = np.hstack(
                                (np.reshape(entity_array1, [entity_array1.shape[0], 1]), results))
                            results_with_id = results_with_id[np.argsort(results_with_id[:, 1])]
                            results_with_id = results_with_id[:, 0].astype(int)
                            #print ("results_with_id : ", results_with_id, "\n")
                            #=[38498 20357 7786 ... .... 27602 1049]
                            #print ("len(results_with_id) : ", len(results_with_id)) 
                            #= 40960 fixed
                            


                            
                            _filter = 0
                            if head_or_tail == 'head':
                                for tmpHead in results_with_id:
                                    #if tmpHead == x_batch[i][0]:
                                        #break
                                    tmpTriple = (tmpHead, x_batch[i][1], x_batch[i][2])
                                    if (tmpTriple in train) or (tmpTriple in valid) or (tmpTriple in test):
                                        continue
                                    else:
                                        _filter += 1
                            else:
                                for tmpTail in results_with_id:
                                    if tmpTail == x_batch[i][2]:
                                        f_errr.write(str(tmpTail) +",")
                                        break
                                    tmpTriple = (x_batch[i][0], x_batch[i][1], tmpTail)
                                    if (tmpTriple in train) or (tmpTriple in valid) or (tmpTriple in test):
                                        continue
                                    else:
                                        _filter += 1
                                        f_errr.write(str(tmpTail) +",")
                            
                            mr += (_filter + 1)
                            mr_temp += (_filter + 1)
                            #print ("mr : ", mr, "\n")
                            #= 3185.0
                            f_errr_rank.write(str(mr_temp)+"\n") #added
                            f_errr.write("\n") #added

                            #mrr += 1.0 / (_filter + 1)
                            #if _filter < 10:
                            #    hits10 += 1

                            end = time.clock()
                            elapsed = end-begin
                            print("use time : " +str(elapsed) +" second")

                        f_errr.close()
                        f_errr_rank.close()



                        return "0"

                    if FLAGS.testIdx < (FLAGS.num_splits - 1):
                        batch_test = 447, FLAGS.testIdx = 0
                        head_results = test_prediction_left(x_test[batch_test * FLAGS.testIdx : batch_test * (FLAGS.testIdx + 1)],
                                                      y_test[batch_test * FLAGS.testIdx : batch_test * (FLAGS.testIdx + 1)],
                                                      head_or_tail='head')
                        tail_results = test_prediction_right(x_test[batch_test * FLAGS.testIdx : batch_test * (FLAGS.testIdx + 1)],
                                                       y_test[batch_test * FLAGS.testIdx : batch_test * (FLAGS.testIdx + 1)],
                                                       head_or_tail='tail')
                    else:
                        head_results = test_prediction_left(x_test[batch_test * FLAGS.testIdx : len_test],
                                                      y_test[batch_test * FLAGS.testIdx : len_test],
                                                      head_or_tail='head')
                        tail_results = test_prediction_right(x_test[batch_test * FLAGS.testIdx : len_test],
                                                       y_test[batch_test * FLAGS.testIdx : len_test],
                                                       head_or_tail='tail')

                    #wri = open(_file + '.eval.' + str(FLAGS.testIdx) + '.txt', 'w')

                    #for _val in head_results:
                    #    wri.write(str(_val) + ' ')
                    #wri.write('\n')
                    #for _val in tail_results:
                        #wri.write(str(_val) + ' ')
                    #wri.write('\n')

                   # wri.close()

