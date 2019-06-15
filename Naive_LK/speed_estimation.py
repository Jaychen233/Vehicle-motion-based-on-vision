'''
Speed estimation using monocular optical flow
=============================================

We estimate the speed of a monocular camera traveling on a road.
The assumption is that the observed optical flow will correspond
to the one induced by a translating camera that's observing a
plane (i.e. road).
'''

import numpy as np
import cv2
import json
from time import clock
from IPython import embed

from matplotlib import pyplot as plt
import pandas as pd

# Settings for the LK tracker and the corner detector
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.1,
                       minDistance = 7,
                       blockSize = 5 )

# Image crops from left and top edges
crop_left = 0
crop_top = 320


class App:
    def __init__(self, video_src):
        self.track_len = 10
        self.detect_interval = 1
        self.tracks = []
        self.cap = cv2.VideoCapture(video_src)
        self.frame_idx = 0

        # Load ground truth
        gt_raw = np.array(json.load(open('drive.json')))

        # Holds data for each image frame
        self.data_table = pd.DataFrame.from_dict({
            'time': gt_raw[:, 0],
            'gt': gt_raw[:, 1],
            'median': np.nan,
            'mean': np.nan,
            'median_h': np.nan,
            'median_v': np.nan,
            'mean_h': np.nan,
            'mean_v': np.nan})

    def run(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = frame[crop_top:440:, crop_left:550]

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            vis = frame.copy()

            #
            # Optical flow tracking

            if len(self.tracks) > 0:
                img0, img1 = self.prev_gray, frame_gray
                # Compute optical flow
                p0 = self.tracks
                p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                # Backtracking check
                d = abs(p0-p0r).reshape(-1, 2).max(-1)
                good = d < 1e9

                # Some visualization
                for (x0, y0), (x1, y1), good_flag in zip(p0.reshape(-1, 2), p1.reshape(-1, 2), good):
                    cv2.circle(vis, (x1, y1), 2, (0, 255, 0), -1)
                    pl = np.array([[x0, y0], [x1, y1]], dtype=np.int32)
                    cv2.polylines(vis, [pl], 1, (0, 255, 0))

                # Build an array of (x, y, dx, dy) flows
                flows = np.hstack((p0.reshape(-1, 2), (p1 - p0).reshape(-1, 2)))

                # Switch pixel coordinates to a new frame centered in the middle point
                flows[:, 0] += crop_left - 320
                flows[:, 1] += crop_top - 240

                # Compute the V/hf factor for each flow (2 per flow)
                n_flows = flows.shape[0]

                if n_flows:
                    V_hf = np.zeros(n_flows*2)
                    V_hf[:n_flows] = flows[:, 2] / flows[:, 0] / flows[:, 1]  # H flow
                    V_hf[n_flows:] = flows[:, 3] / (flows[:, 1]**2)  # V flow

                    # Store output
                    self.data_table['median'].iloc[self.frame_idx] = np.median(V_hf)
                    self.data_table['median_h'][self.frame_idx]= np.median(V_hf[:n_flows])
                    self.data_table['median_v'][self.frame_idx]  = np.median(V_hf[n_flows:])

                    self.data_table['mean'][self.frame_idx] = np.mean(V_hf)
                    self.data_table['mean_h'][self.frame_idx] = np.mean(V_hf[:n_flows])
                    self.data_table['mean_v'][self.frame_idx] = np.mean(V_hf[n_flows:])
            # Detection of new points
            mask = np.zeros_like(frame_gray)
            mask[:] = 255
            for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                cv2.circle(mask, (x, y), 5, 0, -1)
            # Mask top center (likely covered by the car ahead)
            cv2.rectangle(mask, (200, 0), (440, 50), 0, -1)
            # Mask edges (outside the road plane)
            cv2.fillPoly(mask,
                         np.array([[(345, 0), (frame_gray.shape[1], 65), (frame_gray.shape[1], 0)]]), 0)
            # cv2.imshow('mask', mask)

            p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
            if p is not None:
                self.tracks = p  # p.shape == (n , 1, 2)

            self.frame_idx += 1
            self.prev_gray = frame_gray

            if False:
                cv2.imshow('lk_track', vis)
                ch = 0xFF & cv2.waitKey(1)
                if ch == 27:
                    break

        #
        # Final processing
        self.cap.release()

        # Estimate scale by using the first n frames
        num_training_frames = 1000
        hf_factor = np.linalg.lstsq(self.data_table['median'][1:num_training_frames].values.reshape(-1,1),
                                    self.data_table['gt'][1:num_training_frames].values.reshape(-1, 1))[0][0][0]
        print("Estimated hf factor = {}".format(hf_factor))

#         Use the estimated scale for the rest of the images
        self.data_table[['median', 'median_h', 'median_v', 'mean', 'mean_h', 'mean_v']] *= hf_factor

        # Some low-passing
        self.data_table['median_low'] = self.data_table['median'].rolling(window=30).mean()
        # Plot
        ax = self.data_table[['ground_truth', 'predcited']].plot(figsize=(8, 5))
        ax.axvline(x=num_training_frames, color='k', ls='--')
        ax.set_xlim(0, 4000)
        ax.set_xlabel('Frame num.')
        ax.set_ylabel('Speed [m/s]')

        ax.figure.savefig('result.png', bbox_inches='tight')

        plt.show()

def main():
    import sys
    print(__doc__)

    App('drive.mp4').run()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
