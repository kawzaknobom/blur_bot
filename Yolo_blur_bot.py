from pyrogram import Client,filters
from pyrogram.types import Message
import os,shutil,cv2
from Cookies_File import Admin_Ids,Audio_Forms,Api_Id,Api_Hash
from detection import segment
from PIL import Image
# from Common_Funcs import Pyrogram_Client,Blur_Female,Check_Dir

def Pyrogram_Client(Bot_Token):
  Bot_Identifier = Bot_Token.split(':')[0]
  Session_file = Bot_Identifier+'_session_prm_bot'
  bot = Client(Session_file,api_id=Api_Id,api_hash=Api_Hash,bot_token=Bot_Token)
  return bot,Bot_Identifier

Bot_Token = '8516897868:AAG8tyDbCdDYmcQHyBgnxf_fI5nwDUulcMY'
bot,Bot_Identifier = Pyrogram_Client(Bot_Token)
Dl_Dir = f'./Blur_{Bot_Identifier}/'


def remove_black_background_and_composite(foreground, background, threshold=5):
    datas = foreground.getdata()
    newData = []
    for item in datas:
        if item[0] < threshold and item[1] < threshold and item[2] < threshold:
            newData.append((255, 255, 255, 0)) # Fully transparent
        else:
            newData.append(item) 
            
    foreground.putdata(newData)
    background = background.resize(foreground.size)
    final_image = background.copy().convert("RGBA")
    alpha_mask = foreground.getchannel('A')
    final_image.paste(foreground, (0, 0), alpha_mask)
    return final_image.convert("RGB")

def Check_Dir(Dir):
  if os.path.isdir(Dir):
      shutil.rmtree(Dir)
  Create_Dir(Dir)

def Create_Dir(Dir):
  if not os.path.isdir(Dir):
    Mkdir_Cmd = f'mkdir -p "{Dir}"'
    os.system(Mkdir_Cmd)

def Mp3_Conv(File):
  Mp3_File = ('.' if File.startswith('.') else '') +  File.split('.')[(1 if File[0] == '.' else 0)] + '_Conv.mp3'
  Mp3_Cmd = f'ffmpeg -i "{File}" -q:a 0 -map a "{Mp3_File}" -y'
  os.system(Mp3_Cmd)
  return Mp3_File

def Media_Compress(file_path,Rate=None):
  if file_path.lower().endswith(Audio_Forms) : 
   Res_File = ('.' if file_path.startswith('.') else '') + file_path.split('.')[(1 if file_path[0] == '.' else 0)] + '_Comp.mp3'
   Comp_Cmd = f'ffmpeg -i "{file_path}" -b:a "{Rate}" "{Res_File}" -y '
  else :
    Res_File = ('.' if file_path.startswith('.') else '') + file_path.split('.')[(1 if file_path[0] == '.' else 0)] + '_Comp.mp4'
    Comp_Cmd = f'ffmpeg -i "{file_path}" -c:v libx264 -crf 28 "{Res_File}" -y'
  os.system(Comp_Cmd)
  return Res_File

def Vid_Mk(Vid,Aud):
  Vid_Res = ('.' if Vid.startswith('.') else '') + Vid.split('.')[(1 if Vid[0]=='.' else 0)] + '_Merged.mp4'
  Sub_Cmd = f'ffmpeg -i "{Vid}" -i "{Aud}" -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 "{Vid_Res}" -y'
  os.system(Sub_Cmd)
  return Vid_Res


def Blur_Female(file_path):
  P_Name = ('.' if file_path.startswith('.') else '') + file_path.split('.')[1 if file_path[0] == '.' else 0]
  Ex = file_path.split('.')[-1]
  Res_File = f"{P_Name}_Blurred.{Ex}"
  Aud = Mp3_Conv(file_path)
  # file_path = Media_Compress(file_path)
  cap = cv2.VideoCapture(file_path)
  if not cap.isOpened():
    raise ValueError("Error opening video file")
  fps = cap.get(cv2.CAP_PROP_FPS)
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  out = cv2.VideoWriter(Res_File, fourcc, fps, (width, height))
  while(True):
    ret, frame = cap.read()
    if ret:
      cv2.imwrite("detected_human.jpg", frame)
      frame = segment(0.09,'detected_human.jpg')

      # (humans, _) = detect_model.detectMultiScale(frame, winStride=(4, 4),padding=(0, 0), scale=1.05)
      # for (x,y,w,h) in humans:
      #     cv2.imwrite("detected_human.jpg", frame[y:y+h, x:x+w])
      #     if not is_male('detected_human.jpg'):
      #         frame[y:y+h, x:x+w] = cv2.blur(frame[y:y+h, x:x+w], (51, 51))

      out.write(frame)
    else:
        break 
  cap.release()
  out.release()
  cv2.destroyAllWindows()
  Res_File = Vid_Mk(Res_File,Aud)
  Res_File = Media_Compress(Res_File)
  return Res_File


@bot.on_message(filters.command('start') & filters.private)
async def command1(bot,message):
 if message.chat.id in Admin_Ids : 
   await message.reply('لبقية البوتات \n\n @sunnaybots')

@bot.on_message(filters.private & filters.incoming & ( filters.video | filters.photo))
async def _telegram_file(client, message):
 if message.chat.id in Admin_Ids : 
  Reply = await message.reply('جار العمل ...')
  Media_Path = await message.download(file_name=Dl_Dir)
  if message.video :
   Blurred_Vid = Blur_Female(Media_Path)
   await message.reply_video(Blurred_Vid)
   await Reply.edit_text('تمت ')
  elif message.photo :
    P_Name = ('.' if Media_Path.startswith('.') else '') + Media_Path.split('.')[1 if Media_Path[0] == '.' else 0]
    Ex = Media_Path.split('.')[-1]
    Res_File = f"{P_Name}_Blurred.{Ex}"
    Blurred_Photo = segment(0.09,Media_Path)
    Blurred_Photo.save(Res_File)

    image_Orig = Image.open(Media_Path).convert("RGBA")
    image_mask = Image.open(Res_File).convert("RGBA")
    result_image = remove_black_background_and_composite(image_mask, image_Orig,threshold=20 )
    result_image.save(Res_File)
    await message.reply_photo(Res_File)
  Check_Dir(Dl_Dir)
    

   
bot.run()