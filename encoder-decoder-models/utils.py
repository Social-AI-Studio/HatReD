import os
import torch

import numpy as np
    
class Logger(object):
    def __init__(self,output_dir):
        dirname=os.path.dirname(output_dir)
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        self.log_file=open(output_dir,'w')
        self.infos={}
        
    def append(self,key,val):
        vals=self.infos.setdefault(key,[])
        vals.append(val)

    def log(self,extra_msg=''):
        msgs=[extra_msg]
        for key, vals in self.infos.iteritems():
            msgs.append('%s %.6f' %(key,np.mean(vals)))
        msg='\n'.joint(msgs)
        self.log_file.write(msg+'\n')
        self.log_file.flush()
        self.infos={}
        return msg
        
    def write(self,msg):
        self.log_file.write(msg+'\n')
        self.log_file.flush()
        print(msg)    

    def log_hyperpara(self, opt):
        dic = vars(opt)
        for k,v in dic.items():
            self.write(k + ' : ' + str(v))
            
def caption_collate_fn(batch):
    input_sent  = [item['input_sent']for item in batch]
    output_sents = [item['output_sents'] for item in batch]

    return {
        "input_sent": input_sent,
        "output_sents": output_sents
    }


def feature_collate_fn(batch):
    input_sent  = [item['input_sent']for item in batch]
    output_sents = [item['output_sents'] for item in batch]

    visual_embeds = [item['visual_embeds'] for item in batch]
    visual_embeds = np.stack(visual_embeds, axis=0)
    visual_embeds = torch.from_numpy(visual_embeds)

    return {
        "input_sent": input_sent,
        "output_sents": output_sents,
        "visual_embeds": visual_embeds
    }