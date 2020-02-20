

def gen_gt_file():
    pass

def gen_pred_file(path,vid_pred):
    out_file = open('%s-bboxes.txt' % path, 'w')
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
    pass