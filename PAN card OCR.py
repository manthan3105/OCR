import sys
import pytesseract
import cv2
import numpy as np
import imutils
orignalPAN = input('Enter your PAN number')
orignalDOB = input('Enter your Date of Birth')
ODOB = orignalPAN.lower()

v=30

img = cv2.imread('new2.jpg')
#gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
'''
gray1 = cv2.GaussianBlur(gray,(11,11),4,0,4)
scale_percent = 10  # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

# resize image
resized = cv2.resize(gray1, dim, interpolation=cv2.INTER_AREA)

print('Resized Dimensions : ', resized.shape)

cv2.imshow("Resized image", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

flag = 0
flag1 = 0
flag2 = 0
flag3=0


r = 500.0 / img.shape[1]
dim = (500, int(img.shape[0] * r))
img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray1 = cv2.GaussianBlur(gray,(11,11),4,0,4)
ret,thresh = cv2.threshold(gray1,190,255,0)
contours, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
img = cv2.drawContours(img,contours,-1, (0, 255, 0), 3)

cv2.imshow('wh',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
#mask = np.zeros_like(thresh)


new_img=gray
for contour in contours:
    area = cv2.contourArea(contour)
    #print(area)
    try:
        if area > 20000:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            img = cv2.drawContours(img, [box], -1, (0, 255, 0), 3)
            x,y,w,h, = cv2.boundingRect(contour)



            print('width,',w,'  height,',h)
            if w<h:
                print('###########################################################')
                rotated = imutils.rotate_bound(img, 270)
                cv2.imshow("Rotated (Correct)", rotated)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            if w<h: #if rotation needs to be done
                new_img = rotated[y:y+h,x:x+w]
            else:
                new_img = gray[y:y+h,x:x+w]
            cv2.imshow('partial image', new_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    except:
        new_img=gray


height,width = new_img.shape[:2]
start_row,start_col = int(height*0.27), int(width*0)
end_row,end_col = int(height*0.4), int(width*0.9)
cropped =  new_img[start_row:end_row,start_col:end_col]


cv2.imshow('cropped image',cropped)
cv2.waitKey(0)
cv2.destroyAllWindows()
result1 = pytesseract.image_to_string(cropped)





while v<140 and flag3==0:
    retval, threshold = cv2.threshold(gray, v, 255, cv2.THRESH_BINARY)  # 138 #80
    result = pytesseract.image_to_string(threshold,"eng")
    res = result.replace(" ", "")
    res1 = res.replace("\n", "")
    res2 = res1.lower()
    a = len(res2)
    k = 0
    while k + 10 < a and flag2 == 0: #loop to find name
        name_position = result.find('/ Name')
        if name_position != -1:
            flag2=1
            result2=result
            break
        if name_position == -1:
            name_position = result.find('/NAME')
            if name_position != -1:
                flag2 = 1
                result2 = result
                break
        if name_position == -1:
            name_position = result.find('| Name')
            if name_position != -1:
                flag2 = 1
                result2 = result
                break

        print("name wait")
        break
        k = k + 1

    j=0
    while j+10 < a and flag==0: #loop to find PAN
        if 96 < ord(res2[j]) < 123:
            if 96 < ord(res2[j + 1]) < 123:
                if 96 < ord(res2[j + 2]) < 123:
                    if 96 < ord(res2[j + 3]) < 123:
                        if 96 < ord(res2[j + 4]) < 123:
                            if 47 < ord(res2[j + 5]) < 58:
                                if 47 < ord(res2[j + 6]) < 58:
                                    if 47 < ord(res2[j + 7]) < 58:
                                        if 47 < ord(res2[j + 8]) < 58:
                                            if 96 < ord(res2[j + 9]) < 123:
                                                PAN = res2[j:(j + 10)]
                                                #print(PAN)
                                                #print(ODOB)
                                                PAN1=PAN.upper()
                                                if PAN == ODOB:
                                                    print('PAN NO. MATCHESSSSS    ',PAN1)
                                                    flag=1
                                                else:
                                                    print(PAN)
                                                    print("PAN wait")
        if flag==1:
            break
        j = j + 1
    i=0
    while i+10 < a and flag1==0:  #loop to find DOB
        if 46 < ord(res2[i]) < 58:
            if 46 < ord(res2[i + 1]) < 58:
                if ord(res2[i + 2]) == 47 or ord(res2[i + 2]) == 45:
                    if 46 < ord(res2[i + 3]) < 58:
                        if 46 < ord(res2[i + 4]) < 58:
                            if ord(res2[i + 5]) == 47 or ord(res2[i + 5]) == 45:
                                if 46 < ord(res2[i + 6]) < 58:
                                    if 46 < ord(res2[i + 7]) < 58:
                                        if 46 < ord(res2[i + 8]) < 58:
                                            if 46 < ord(res2[i + 9]) < 58:
                                                date = res1[i:(i + 10)]
                                                if orignalDOB == date:
                                                    print('DATE OF BIRTH MATCHESSSSSS    ',date)
                                                    flag1=1
                                                else:
                                                    print(date)
                                                    print("DOB wait")
        if flag1==1:
            break
        i=i+1



    if flag==1 and flag1==1 and flag2==1:
        flag3=1
        break
    v=v+1
    print('threshold=',v)


if flag==0:
    print('PAN does not match')
if flag1==0:
    print('DOB does not match')
if flag2==0: #taking out name for old format
    print('Name is: ',result1)

if flag2==1: #taking out name for new format
    print('position of name is',name_position)
    print('Name is ',result2[(name_position+6):(name_position+20)])


cv2.imshow('orignal image',img)
cv2.waitKey(100)
cv2.imshow('grayscaled image',gray)
cv2.waitKey(100)
cv2.imshow('B&W image',threshold)
cv2.waitKey(0)
cv2.destroyAllWindows()
