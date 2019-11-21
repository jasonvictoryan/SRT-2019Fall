import cv2
 
 
def discern(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cap = cv2.CascadeClassifier(
        r"./cascade.xml"
    )
    Rects = cap.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=3, minSize=(30, 30))
    if len(Rects):
        for Rect in Rects:
            x, y, w, h = Rect
            cv2.rectangle(img, (x, y), (x + h, y + w), (0, 255, 0), 2)
    cv2.imshow("Image", img)
 
 
cap = cv2.VideoCapture(0)
while (1):
    ret, img = cap.read()
    # cv2.imshow("Image", img)
    discern(img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()