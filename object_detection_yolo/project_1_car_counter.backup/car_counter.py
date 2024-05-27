import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import*

cap = cv2.VideoCapture("/home/eldar/PycharmProjects/object_detection_yolo/videos/cars.mp4")

# pravimo yolo model:
model = YOLO("../yolo_weights/yolov8n.pt")

mask = cv2.imread("mask_fix.png")

# Tracking

tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3) # default vrijednost

# 300 550 605 585
line = [200,570,605,585]
totalCount= []
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




while True :
    success, img = cap.read() # spreamo sliku uzetu sa nase web kamere
    #mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

    imgRegion = cv2.bitwise_and(img,mask) # postavljamo regiju i nju prosljedjujemo u stream da prati

    # imgGraphics = cv2.imread("ime_slike.png", cv2.IMREAD_UNCHANGED)
    # img = cvzone.overlayPNG(img,imgGraphics,(0,0))

    results = model(imgRegion, stream = True) # uzimamo rezultate naseg modela koje cemo ubaciti na nasu

    detections = np.empty((0,5))

    for r in results:
        boxes = r.boxes # svi okviri uzeti iz predikcije nase kamere
        for box in boxes: # za svaki okvir sa slike

            # Bounding Box: ##################################################################################

            # ovo je drugi nacin (radi puno bolje) :
            x1, y1, x2, y2 = box.xyxy[0] # spremamo velicine okvira u odvojene varijable
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # potrebno je konvertovati u int
            w, h = x2 - x1, y2 - y1 # width , height okvira

            bbox = int(x1),int(y1),int(w),int(h)

            #cvzone.cornerRect(img,bbox,l=10,t=4, rt=1 , colorR=(255,0,255),colorC=(0,255,0))
            # l -> predstavlja velicinu oznacenih uglova
            # t - > predstavlja debljunu oznacenih uglova
            # rt -> predstavlja debljinu cijelog okvira
            # colorR -> predstavlja boju cijelog okvira (rectangle)
            # colorC -> predstavlja boju uglova (corner)

            # Confidence : #########################################################cvzone.putTextRect(img,f'{conf}',(max(0,x1),max(35,(y1-20))))#########################

            conf = math.ceil((box.conf[0]*100))/100 # postotci koliko je okvir tacan pri prikazu
            #print(conf)

            # izvrsavamo prikaz postotka okvira na samoj slici :

            #cvzone.putTextRect(img,f'{conf}',(max(0,x1),max(35,(y1-20))))
            # linijom iznad izvrsavamo prikaz postotka tacnosti okvira, a drugi argument predstavlja
            # poziciju gdje zelimo da bude prikaz postotka
            # max(nula, ili ova vrijednost) ovo je da postotak ostane na ekranu


            # Class Name : #############################################################################
            index_name  = math.ceil(box.cls[0])
            name = names[index_name]
            if conf > 0.2 and (name == "auto" or name == "motocikl" or name == "autobus" or name == "kamion") :
                #cvzone.cornerRect(img, bbox, l=10, t=4, rt=4, colorR=(255, 0, 255), colorC=(0, 255, 0))
                #cvzone.putTextRect(img, f'{name}:{conf}', (max(0, x1), max(35, (y1 - 7))),
                 #                  scale=0.7, thickness=1, offset=3)
                currentArray = np.array([x1,y1,x2,y2,conf]) # ovo je potrebno zbog tracking-a
                detections = np.vstack((detections, currentArray)) # zajedno sa ovim

            # offset prestavlja velicinu boxa u kojem su ispisani podaci





    resultsTracker = tracker.update(detections)

    cv2.line(img,(line[0],line[1]),(line[2],line[3]), (0,0,255) ,5) # zadnja 2 arg su : color, thickness

    for result in resultsTracker:
        x1,y1,x2,y2,Id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w , h = y2 - y1 , x2 - x1
        tracker_box = x1,y1,w,h

        cvzone.cornerRect(img, tracker_box, l=10, t=4, rt=2, colorR=(255, 0, 255), colorC=(0, 255, 0))
        cvzone.putTextRect(img, f'{Id}', (max(0, x1), max(35, (y1 - 7))),
                           scale=2, thickness=1, offset=4)

        cx,cy = x1+w//2, y1+h//2
        cv2.circle(img,(cx,cy),5,(255,0,255),cv2.FILLED)

        if line[0] < cx < line[2] and line[1]-25<cy<line[1]+25:
            if totalCount.count(Id) == 0:
                totalCount.append(Id)
                print(totalCount.count(Id))
                cv2.line(img, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), 5)  # zadnja 2 arg su : color, thickness

        print(result)

    cvzone.putTextRect(img, f'Count: {len(totalCount)}', (50,50))
    # cv2.putText(img,str(len(totalCount)),(255,100),cv2.FONT_HERSHEY_PLAIN,5,(50,50,255),8)

    cv2.imshow("Image",img) # slika koju zelimo da prikazemo
    #cv2.imshow("ImageRegion", imgRegion)
    cv2.waitKey(1) # dajemo 1ms deleya
