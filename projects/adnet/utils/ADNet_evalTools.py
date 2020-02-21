from utils.my_util import get_ILSVRC_eval_infos

def gen_gt_file(path):
    videos_infos,train_videos=get_ILSVRC_eval_infos()
    out_file = open('%s-gt.txt' % path, 'w')
    for tj in range(len(videos_infos)):
        for ti in range(len(videos_infos[tj]['gt'])):
            for tk in range(len(videos_infos[tj]['gt'][0])):
                out_file.write(str(tj) + ',' +
                               str(ti) + ',' +
                               str(videos_infos[tj]['trackid'][ti][tk]) + ',' +
                               str(videos_infos[tj]['name'][ti][tk]) + ',' +
                               str('1') + ',' +
                               str(videos_infos[tj]['gt'][ti][tk][0]) + ',' +
                               str(videos_infos[tj]['gt'][ti][tk][1]) + ',' +
                               str(videos_infos[tj]['gt'][ti][tk][2]) + ',' +
                               str(videos_infos[tj]['gt'][ti][tk][3]) + '\n')
    out_file.close()


def gen_pred_file(path,vid_pred):
    out_file = open('%s-pred.txt' % path, 'w')
    for ti in range(len(vid_pred['bbox'])):
        out_file.write(str(vid_pred['vid_id']) + ',' +
                       str(vid_pred['frame_id'][ti]) + ',' +
                       str(vid_pred['track_id'][ti]) + ',' +
                       str(vid_pred['obj_name'][ti]) + ',' +
                       str(vid_pred['score_cls'][ti]) + ',' +
                       str(vid_pred['bbox'][ti][0]) + ',' +
                       str(vid_pred['bbox'][ti][1]) + ',' +
                       str(vid_pred['bbox'][ti][2]) + ',' +
                       str(vid_pred['bbox'][ti][3]) + '\n')
    out_file.close()


if __name__ == "__main__":
    gen_gt_file('../datasets/data/ILSVRC-vid-eval')