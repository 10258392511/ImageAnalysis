import numpy as np
import cv2
import os
import re
from helpers.utils import read_and_resize


if __name__ == '__main__':
    def adjust_text_pos(x, y, img_size: tuple, x_offset=50, y_offset=50):
        H, W = img_size[:2]
        if x < 0:
            x += x_offset
        if x + x_offset >= W - 1:
            x -= x_offset
        if y < 0:
            y += y_offset
        if y + y_offset >= H - 1:
            y -= y_offset

        return x, y

    def on_mouse(event, x, y, flags, params):
        img = params.get("img")
        filename = params.get("log", None)
        ind = params.get("index", -1)
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(img, (x, y), 3, (0, 0, 255), -1, lineType=cv2.LINE_AA)
            x_pos, y_pos = adjust_text_pos(x, y, img.shape)
            cv2.putText(img, f"{ind}: ({x}, {y})", (x_pos, y_pos), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.4,
                        color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA, bottomLeftOrigin=False)
            cv2.imshow("img", img)
            if filename is not None:
                with open(filename, "a") as wf:
                    wf.write(f"{x}, {y}\n")
            params["index"] += 1


    all_paths = os.listdir("./imgs")
    frame_paths = []
    pattern = re.compile(r"fr[0-9]*.png")
    for path in all_paths:
        if re.match(pattern, path):
            frame_paths.append(path)

    out_base = "./imgs_out"
    in_base = "./imgs"
    for path in frame_paths:
        read_path = os.path.join(in_base, path)
        delimeter_ind = path.find(".png")
        base = path[:delimeter_ind]
        suffix = path[delimeter_ind:]
        img_resize_path = os.path.join(out_base, f"{base}_orig{suffix}")
        mark_path = os.path.join(out_base, f"{base}_mark{suffix}")
        log_path = os.path.join(out_base, f"{base}_log.txt")

        with open(log_path, "w") as wf:
            pass

        ind = 1
        img = read_and_resize(read_path, fx=0.2, fy=0.2, if_save=True, new_path=img_resize_path)
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.namedWindow("img", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("img", on_mouse, param=dict(img=img_color, log=log_path, index=1))
        cv2.imshow("img", img_color)
        # print(img.shape)
        key = cv2.waitKey(0)

        if key == ord("s"):
            delimiter_ind = path.find(".png")
            save_path = path[:delimiter_ind] + "_out" + path[delimiter_ind:]
            print(img_color.shape)
            # plt.imshow(img_color)
            # plt.show()
            cv2.imwrite(mark_path, (img_color * 255).astype(np.uint8))
            cv2.destroyAllWindows()

        if key == ord("q"):
            cv2.destroyAllWindows()
