from ultralytics import YOLO
import cv2
import cvzone
import math

cap = cv2.VideoCapture(0) # uzimam vido sa moje web camere
#postavljamo zeljenu velicinu prikaza web camere:
cap.set(3,1280) # 3, 640
cap.set(4,720) # 4, 480

#cap = cv2.VideoCapture("BIT_CENTAR/BIT_CENTAR/A09_20240313111031.mp4")

# A09_20240313111031.mp4

# pravimo yolo model:
model = YOLO("../yolo_weights/yolov8l.pt")

names = [
    "osoba", "bicikl", "auto", "motocikl", "avion", "autobus", "voz", "kamion",
    "camac", "semafor", "hidrantska pumpa", "stop znak", "parking metar", "klupa",
    "ptica", "macka", "pas", "konj", "ovca", "krava", "slon", "medvjed", "zebra",
    "zirafa", "ruksak", "kisobran", "torbica", "kravata", "kofer", "frizbi",
    "skije", "snowboard", "sportska lopta", "zmaj", "bejzbol palica", "bejzbol rukavica",
    "skejtbord", "surf daska", "teniski reket", "flasa", "casa za vino", "solja",
    "viljuska", "noz", "kasika", "zdjela", "banana", "jabuka", "sendvic", "narandza",
    "brokula", "mrkva", "hot dog", "pica", "krofna", "kolac", "stolica", "kauc",
    "saksija za cvijece", "krevet", "trpezarijski sto", "toalet", "TV", "laptop", "mis",
    "daljinski", "tastatura", "mobilni telefon", "mikrovalna pecnica", "rerna", "toster", "sudoper",
    "frizider", "knjiga", "sat", "vaza", "makaze", "plisani medvjedic", "fen za kosu",
    "cetkica za zube"
]


"""
classNames = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", 
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", 
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", 
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", 
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", 
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", 
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", 
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", 
    "potted plant", "bed", "dining table", "toilet", "TV", "laptop", "mouse", 
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", 
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", 
    "toothbrush"
]
"""

while True :
    success, img = cap.read() # spreamo sliku uzetu sa nase web kamere
    results = model(img, stream = True) # uzimamo rezultate naseg modela koje cemo ubaciti na nasu kameru
    for r in results:
        boxes = r.boxes # svi okviri uzeti iz predikcije nase kamere
        for box in boxes: # za svaki okvir sa slike
            # ovo ispod sto je zakomentarisano radi super
            #x1,y1,x2,y2 = box.xyxy[0] # xyxy ili xywh (bolje je xyxy)
            #x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)
            #cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)


            # Bounding Box: ##################################################################################

            # ovo je drugi nacin (radi puno bolje) :
            x1, y1, x2, y2 = box.xyxy[0] # spremamo velicine okvira u odvojene varijable
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # potrebno je konvertovati u int
            w, h = x2 - x1, y2 - y1 # width , height okvira

            #x1, y1, w, h = box.xywh[0] # iz nekog razloga ovo ne radi ...

            bbox = int(x1),int(y1),int(w),int(h)
            cvzone.cornerRect(img,bbox,l=30,t=4, rt=1 , colorR=(255,0,255),colorC=(0,255,0))

            # l -> predstavlja velicinu oznacenih uglova
            # t - > predstavlja debljunu oznacenih uglova
            # rt -> predstavlja debljinu cijelog okvira
            # colorR -> predstavlja boju cijelog okvira (rectangle)
            # colorC -> predstavlja boju uglova (corner)

            # Confidence : #########################################################cvzone.putTextRect(img,f'{conf}',(max(0,x1),max(35,(y1-20))))#########################

            conf = math.ceil((box.conf[0]*100))/100 # postotci koliko je okvir tacan pri prikazu
            print(conf)

            # izvrsavamo prikaz postotka okvira na samoj slici :

            #cvzone.putTextRect(img,f'{conf}',(max(0,x1),max(35,(y1-20))))
            # linijom iznad izvrsavamo prikaz postotka tacnosti okvira, a drugi argument predstavlja
            # poziciju gdje zelimo da bude prikaz postotka
            # max(nula, ili ova vrijednost) ovo je da postotak ostane na ekranu


            # Class Name : #############################################################################
            index_name  = math.ceil(box.cls[0])
            cvzone.putTextRect(img, f'{names[index_name]}:{conf}', (max(0, x1), max(35, (y1 - 20))), scale=1,thickness=1)





    cv2.imshow("Image",img) # slika koju zelimo da prikazemo
    cv2.waitKey(1) # dajemo 1ms deleya
