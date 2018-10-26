#!/usr/bin/env python

from __future__ import print_function

import argparse
import os, codecs
import time, datetime


from six.moves import cPickle


from six import text_type

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

parser = argparse.ArgumentParser(
                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--save_dir', type=str, default='save',
                    help='model directory to store checkpointed models')
parser.add_argument('-n', type=int, default=500,
                    help='number of characters to sample')
parser.add_argument('--prime', type=text_type, default=u'',
                    help='prime text')
parser.add_argument('--sample', type=int, default=1,
                    help='0 to use max at each timestep, 1 to sample at '
                         'each timestep, 2 to sample on spaces')
parser.add_argument('--name', type=str, default='')

args = parser.parse_args()

import tensorflow as tf
from model import Model

def sample(args):
    start = time.time()
    if args.name is not None:
        save_dir = os.path.join(args.save_dir,args.name)
    with open(os.path.join(save_dir, 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)
    with open(os.path.join(save_dir, 'chars_vocab.pkl'), 'rb') as f:
        chars, vocab = cPickle.load(f)
    #Use most frequent char if no prime is given
    if args.prime == '':
        args.prime = chars[0]
    model = Model(saved_args, training=False)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            # m = model.sample(sess, chars, vocab, args.n, args.prime,
                            #   args.sample)
            # e = m.encode('utf-8')
            # s = bytes.decode(e)
            # s = bytes.decode(model.sample(sess, chars, vocab, args.n, args.prime,
            #                   args.sample).encode('utf-8'))
            # s = (model.sample(sess, chars, vocab, args.n, args.prime,
            #     args.sample)).decode('utf-8')
            # s = s[2:-1]
            
            # print(s.encode('utf-8'),end='\n\n')
            s = bytes.decode(model.sample(sess, chars, vocab, args.n, args.prime,
                               args.sample).encode('ascii',errors='ignore'),errors='ignore')

            while '\\n' in s:
                i = s.find('\\n')
                s = s[:i]+'\n'+s[i+2:]
            while '\\t' in s:
                i = s.find('\\t')
                s = s[:i]+'\t'+s[i+2:]
            while '\\\'' in s:
                i = s.find('\\\'')
                s = s[:i]+'\''+s[i+2:]
            while '\\r' in s:
                i = s.find('\\r')
                s = s[:i]+'\r'+s[i+2:]
            while '\\xe2\\x80\\x99' in s:
                i = s.find('\\xe2\\x80\\x99')
                s = s[:i]+ '\'' +s[i+12:]
            print(s)
            # print(s.encode('utf-8'))
            with codecs.open(args.save_dir+'\\'+args.name+" sample.txt","w",encoding = "utf-8") as f:
                f.write(s)
    #with codecs.open("output.txt",'r',encoding='utf-8') as f:
    #    for line in f:
    #        print(line,end="")
    end = time.time()
    print('\nTime to complete: {}'.format(datetime.timedelta(seconds=(end-start),microseconds=0))[:-4])
    print('{0:.1f} char/sec'.format(args.n/(end-start)),end='')

if __name__ == '__main__':
    sample(args)
