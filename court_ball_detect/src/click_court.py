# tools/click_court.py
import cv2, json, argparse, os
import numpy as np

pts = []
show_grid = True

def draw_points(img, pts):
    for i,(x,y) in enumerate(pts):
        cv2.circle(img, (int(x),int(y)), 6, (0,255,0), -1)
        cv2.putText(img, f"{i+1}", (int(x)+8,int(y)-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)
    if len(pts) >= 2:
        for i in range(1, len(pts)):
            cv2.line(img, (int(pts[i-1][0]),int(pts[i-1][1])),
                          (int(pts[i][0]),int(pts[i][1])), (0,255,0), 2)

def world_grid(W=18.0, H=9.0, step=1.0):
    lines = []
    xs = np.arange(0, W+1e-6, step)
    ys = np.arange(0, H+1e-6, step)
    for x in xs:
        lines.append(np.array([[x,0],[x,H]], dtype=np.float32))
    for y in ys:
        lines.append(np.array([[0,y],[W,y]], dtype=np.float32))
    return lines

def overlay_preview(frame, img_quad, H, Wm, Hm, draw_grid=True):
    vis = frame.copy()
    # 外框
    cv2.polylines(vis, [img_quad.astype(np.int32)], isClosed=True, color=(0,255,255), thickness=2)
    # 網格（把世界直線投回影像）
    if draw_grid and H is not None:
        Hinv = np.linalg.inv(H)
        for seg in world_grid(Wm, Hm, step=1.0):
            pts_img = cv2.perspectiveTransform(seg.reshape(-1,1,2), Hinv).reshape(-1,2)
            p0 = tuple(pts_img[0].astype(int)); p1 = tuple(pts_img[1].astype(int))
            cv2.line(vis, p0, p1, (0,128,255), 1, cv2.LINE_AA)
    return vis

def on_mouse(event, x, y, flags, param):
    global pts
    if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
        pts.append((float(x), float(y)))

def grab_frame(src, frame_idx=None, time_s=None):
    # 先當作圖片讀
    if os.path.isfile(src):
        img = cv2.imread(src)
        if img is not None:
            return img

    # 當作影片讀：嘗試多個後端（較能吃 .mov/HEVC）
    apis = []
    for name in ("CAP_FFMPEG","CAP_MSMF","CAP_DSHOW","CAP_ANY"):
        if hasattr(cv2, name):
            apis.append(getattr(cv2, name))

    for api in apis:
        cap = cv2.VideoCapture(src, api)
        if not cap.isOpened():
            cap.release()
            continue
        if frame_idx is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        elif time_s is not None:
            cap.set(cv2.CAP_PROP_POS_MSEC, time_s*1000.0)
        ok, frame = cap.read()
        cap.release()
        if ok and frame is not None:
            print(f"[INFO] opened source with API={api}")
            return frame

    raise RuntimeError(f"Cannot open source: {src}")

def main():
    global pts, show_grid
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="video file or image file")
    ap.add_argument("--out", default="court.json")
    ap.add_argument("--w", type=float, default=18.0, help="court width in meters")
    ap.add_argument("--h", type=float, default=9.0, help="court height in meters")
    ap.add_argument("--line_w", type=float, default=0.05)
    ap.add_argument("--frame", type=int, default=None, help="frame index to pick (for video)")
    ap.add_argument("--time", type=float, default=None, help="timestamp seconds (for video)")
    args = ap.parse_args()

    frame = grab_frame(args.video, args.frame, args.time)
    base = frame.copy()

    cv2.namedWindow("click 4 corners: TL -> TR -> BR -> BL", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("click 4 corners: TL -> TR -> BR -> BL",
                     min(1280, frame.shape[1]), min(720, frame.shape[0]))
    cv2.setMouseCallback("click 4 corners: TL -> TR -> BR -> BL", on_mouse)

    H = None
    img_quad = None

    print("請依序點四個『外框角』：左上 → 右上 → 右下 → 左下")
    print("快捷鍵：Z=復原, C=清空, G=切換網格, S=儲存, Q/ESC=離開")

    while True:
        vis = base.copy()

        # 指示
        msg = "Click OUTER corners: TL, TR, BR, BL  (Z=undo, C=clear, G=grid, S=save, Q=quit)"
        cv2.putText(vis, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

        # 畫點與線
        draw_points(vis, pts)

        # 若已 4 點：計算 H 並做預覽
        if len(pts) == 4:
            img_pts = np.array(pts, dtype=np.float32).reshape(4,1,2)
            world_pts = np.array([[0,0],[args.w,0],[args.w,args.h],[0,args.h]], dtype=np.float32).reshape(4,1,2)
            H, _ = cv2.findHomography(img_pts, world_pts, method=cv2.RANSAC)
            if H is not None:
                # 用世界外框投回影像，避免點序小誤差導致亂線
                world_rect = np.array([[0,0],[args.w,0],[args.w,args.h],[0,args.h]], dtype=np.float32).reshape(4,1,2)
                img_rect = cv2.perspectiveTransform(world_rect, np.linalg.inv(H))
                vis = overlay_preview(vis, img_rect, H, args.w, args.h, draw_grid=show_grid)
                img_quad = img_rect.copy()

                # 顯示簡單的重投影誤差
                proj = cv2.perspectiveTransform(img_pts, H)  # to world
                back = cv2.perspectiveTransform(proj, np.linalg.inv(H))  # back to img
                err = float(np.mean(np.linalg.norm(back.reshape(-1,2) - img_pts.reshape(-1,2), axis=1)))
                cv2.putText(vis, f"reproj err: {err:.2f}px", (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)

        cv2.imshow("click 4 corners: TL -> TR -> BR -> BL", vis)
        k = cv2.waitKey(20) & 0xFF
        if k in (27, ord('q')):  # ESC / q
            break
        elif k in (ord('z'), ord('Z')) and len(pts)>0:
            pts.pop()
        elif k in (ord('c'), ord('C')):
            pts.clear(); H = None; img_quad = None
        elif k in (ord('g'), ord('G')):
            show_grid = not show_grid
        elif k in (ord('s'), ord('S')):
            if len(pts) != 4 or H is None:
                print("Need exactly 4 points and a valid homography.")
                continue
            img_pts_out = np.array(pts, dtype=float).tolist()
            cfg = {
                "world_court": {
                    "width_m": float(args.w),
                    "height_m": float(args.h),
                    "line_width_m": float(args.line_w),
                    "outer_lines_included": True
                },
                "homography": {
                    "image_points": img_pts_out,
                    "world_points": [[0.0,0.0],[float(args.w),0.0],[float(args.w),float(args.h)],[0.0,float(args.h)]]
                },
                "mask_polygon_image": img_pts_out
            }
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
            print(f"[OK] saved -> {os.path.abspath(args.out)}")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
