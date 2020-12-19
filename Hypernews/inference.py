from Hypernews.model import MODEL
import tensorflow as tf
import numpy as np
import datetime
from Hypernews.data_loader import load_data


def get_feed_dict(model, data, start, end):
    feed_dict = {model.news_title: data.news_title[start:end],
                 model.clicked_titles: data.clicked_titles[start:end],
                 model.active_time: data.active_time[start:end],
                 model.labels: data.labels[start:end],
                 model.time_label: data.time_label[start:end],
                 model.news_category: data.news_category[start:end],
                 model.user_city: data.user_city[start:end],
                 model.user_region: data.user_region[start:end],
                 model.clicked_category: data.clicked_category[start:end],
                 model.news_age: data.news_age[start:end],
                 model.news_id: data.news_id[start:end],
                 model.news_len: data.news_len[start:end],
                 model.clicked_news_len: data.clicked_news_len[start:end],
                 model.clicked_news_id: data.clicked_news_id[start:end],
                 model.is_train: data.is_train,
                 model.weights: data.weights[start:end]}
    return feed_dict


def inference(args):

    model = MODEL(args)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        
        print('loading model..')
        saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
        saver.restore(sess, 'model/mymodel.ckpt-6')

        print('loading data..')
        
        load_file = args.file_directory+str(args.first_train_file)
        train_data = load_data(args, load_file)

        load_file = args.file_directory+str(args.first_test_file)
        test_data = load_data(args, load_file)

        for step in range(1):

            args.is_train = False

            train_loss = 0
            train_time_loss = 0
            train_batchs = 0
            train_labels = []
            train_scores = []
            train_time_true = []
            train_time_pre = []
            for i in range(args.train_file_num):
                #load_file=args.file_directory+str(args.first_train_file+i)
                #train_data=load_data(args,load_file)
                start_list = list(range(0, train_data.size, args.batch_size*10))
                train_batchs = train_batchs + len(start_list)
                for start in start_list:
                    end = start + args.batch_size*10
                    loss, labels, scores, time_loss, time_pre, time_true, time_label = model.test(sess, get_feed_dict(model, train_data, start, end))
                    train_labels.extend(labels)
                    train_scores.extend(scores)
                    for k in range(len(time_label)):
                        if time_label[k] == 0:
                            continue
                        train_time_true.append(time_true[k])
                        train_time_pre.append(time_pre[k])
                    train_loss=train_loss+loss
                    train_time_loss = train_time_loss + time_loss

            train_auc, train_f1, train_pre, train_recall, train_time_f1 = model.eval(args, train_labels, train_scores, train_time_true, train_time_pre)
            train_loss = train_loss / train_batchs
            train_time_loss = train_time_loss / train_batchs

            test_loss = 0
            test_time_loss = 0
            test_batchs = 0
            test_labels = []
            test_scores = []
            test_time_true = []
            test_time_pre = []
            for i in range(args.test_file_num):
                #load_file=args.file_directory+str(args.first_test_file+i)
                #test_data=load_data(args,load_file)
                start_list = list(range(0, test_data.size, args.batch_size*10))
                test_batchs = test_batchs + len(start_list)
                for start in start_list:
                    end = start + args.batch_size*10
                    loss, labels, scores, time_loss, time_pre, time_true, time_label = model.test(sess, get_feed_dict(model, test_data, start, end))
                    test_labels.extend(labels)
                    test_scores.extend(scores)
                    for k in range(len(time_label)):
                        if time_label[k] == 0:
                            continue
                        test_time_true.append(time_true[k])
                        test_time_pre.append(time_pre[k])
                    test_loss=test_loss+loss
                    test_time_loss  = test_time_loss + time_loss
            test_auc, test_f1, test_pre, test_recall, test_time_f1 = model.eval(args, test_labels, test_scores, test_time_true, test_time_pre)
            test_loss = test_loss / test_batchs
            test_time_loss = test_time_loss / test_batchs

            time_stamp=datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S')
            #log_str='time:'+time_stamp+'  epoch %d train_loss:%.4f test_loss:%.4f train_auc:%.4f test_auc:%.4f train_f1_score:%.4f test_f1_score:%.4f' % (step, train_loss,test_loss,train_auc,test_auc,train_f1,test_f1)
            log_str = 'time:'+time_stamp+'  epoch %d \ntrain_loss:%.4f test_loss:%.4f train_time_loss:%.4f test_time_loss:%.4f train_time_f1:%.4f test_time_f1:%.4f \n' \
                                       'train_auc:%.4f test_auc:%.4f train_f1_score:%.4f test_f1_score:%.4f train_pre:%.4f test_pre:%.4f train_recall:%.4f test_recall:%.4f '\
                                       % (step, train_loss, test_loss, train_time_loss, test_time_loss, train_time_f1, test_time_f1, train_auc, test_auc, train_f1, test_f1, train_pre, test_pre, train_recall, test_recall)
            print(log_str)

