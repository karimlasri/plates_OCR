import sys
from detect_chars import detect_characters

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage : python main.py <img_path> <dest_folder> <show_images=False>")
    else:
        img_path = sys.argv[1]
        dest_folder = sys.argv[2]
        show = False
        if len(sys.argv) >= 4:
            if sys.argv[3] == '1':
                show = True
        detect_characters(img_path, dest_folder, show)