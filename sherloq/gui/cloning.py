from itertools import compress
from os.path import splitext
from time import time

class CloningWidget(ToolWidget):
    def __init__(self, image):
        self.image = image
        self.gray = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        self.total = self.kpts = self.desc = self.matches = self.clusters = self.mask = None
        self.detector_combo = 'BRISK'
        self.response_spin = 90
        self.matching_spin = 20
        self.distance_spin = 15
        self.cluster_spin = 5
        self.nolines_check = True
        self.kpts_check = True
        self.mask = None


    def process(self):
        algorithm = self.detector_combo
        response = 100 - self.response_spin
        matching = self.matching_spin.value() / 100 * 255
        distance = self.distance_spin.value() / 100
        cluster = self.cluster_spin.value()

        if self.kpts is None:
            if algorithm == 0:
                detector = cv.BRISK_create()
            elif algorithm == 1:
                detector = cv.ORB_create()
            elif algorithm == 2:
                detector = cv.AKAZE_create()
            else:
                return

            mask = self.mask
            self.kpts, self.desc = detector.detectAndCompute(self.gray, mask)
            self.total = len(self.kpts)
            responses = np.array([k.response for k in self.kpts])
            strongest = (cv.normalize(responses, None, 0, 100, cv.NORM_MINMAX) >= response).flatten()
            self.kpts = list(compress(self.kpts, strongest))
            if len(self.kpts) > 30000:
                self.kpts = self.desc = None
                self.total = 0
                self.desc = self.desc[strongest]

        if self.matches is None:
            matcher = cv.BFMatcher_create(cv.NORM_HAMMING, True)
            self.matches = matcher.radiusMatch(self.desc, self.desc, matching)
            self.matches = [item for sublist in self.matches for item in sublist]
            self.matches = [m for m in self.matches if m.queryIdx != m.trainIdx]

        if not self.matches:
            self.clusters = []
        elif self.clusters is None:
            self.clusters = []
            min_dist = distance * np.min(self.gray.shape) / 2
            kpts_a = np.array([p.pt for p in self.kpts])
            ds = np.linalg.norm([kpts_a[m.queryIdx] - kpts_a[m.trainIdx] for m in self.matches], axis=1)
            self.matches = [m for i, m in enumerate(self.matches) if ds[i] > min_dist]

            total = len(self.matches)
            for i in range(total):
                match0 = self.matches[i]
                d0 = ds[i]
                query0 = match0.queryIdx
                train0 = match0.trainIdx
                group = [match0]

                for j in range(i + 1, total):
                    match1 = self.matches[j]
                    query1 = match1.queryIdx
                    train1 = match1.trainIdx
                    if query1 == train0 and train1 == query0:
                        continue
                    d1 = ds[j]
                    if np.abs(d0 - d1) > min_dist:
                        continue

                    a0 = np.array(self.kpts[query0].pt)
                    b0 = np.array(self.kpts[train0].pt)
                    a1 = np.array(self.kpts[query1].pt)
                    b1 = np.array(self.kpts[train1].pt)

                    aa = np.linalg.norm(a0 - a1)
                    bb = np.linalg.norm(b0 - b1)
                    ab = np.linalg.norm(a0 - b1)
                    ba = np.linalg.norm(b0 - a1)

                    if not (0 < aa < min_dist and 0 < bb < min_dist or 0 < ab < min_dist and 0 < ba < min_dist):
                        continue
                    for g in group:
                        if g.queryIdx == train1 and g.trainIdx == query1:
                            break
                    else:
                        group.append(match1)

                if len(group) >= cluster:
                    self.clusters.append(group)
                
        output = np.copy(self.image)
        hsv = np.zeros((1, 1, 3))
        nolines = self.nolines_check
        show_kpts = self.kpts_check

        if show_kpts:
            for kpt in self.kpts:
                cv.circle(output, (int(kpt.pt[0]), int(kpt.pt[1])), 2, (250, 227, 72))

        angles = []
        for c in self.clusters:
            for m in c:
                ka = self.kpts[m.queryIdx]
                pa = tuple(map(int, ka.pt))
                sa = int(np.round(ka.size))
                kb = self.kpts[m.trainIdx]
                pb = tuple(map(int, kb.pt))
                sb = int(np.round(kb.size))
                angle = np.arctan2(pb[1] - pa[1], pb[0] - pa[0])
                if angle < 0:
                    angle += np.pi
                angles.append(angle)
                hsv[0, 0, 0] = angle / np.pi * 180
                hsv[0, 0, 1] = 255
                hsv[0, 0, 2] = m.distance / matching * 255
                rgb = cv.cvtColor(hsv.astype(np.uint8), cv.COLOR_HSV2BGR)
                rgb = tuple([int(x) for x in rgb[0, 0]])
                cv.circle(output, pa, sa, rgb, 1, cv.LINE_AA)
                cv.circle(output, pb, sb, rgb, 1, cv.LINE_AA)
                if not nolines:
                    cv.line(output, pa, pb, rgb, 1, cv.LINE_AA)

        regions = 0
        if angles:
            angles = np.reshape(np.array(angles, dtype=np.float32), (len(angles), 1))
            if np.std(angles) < 0.1:
                regions = 1
            else:
                criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                attempts = 10
                flags = cv.KMEANS_PP_CENTERS
                compact = [cv.kmeans(angles, k, None, criteria, attempts, flags)[0] for k in range(1, 11)]
                compact = cv.normalize(np.array(compact), None, 0, 1, cv.NORM_MINMAX)
                regions = np.argmax(compact < 0.005) + 1

        return output