import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
import time
import subprocess 
import win32gui
import win32process
import keyboard_nofocus
import time
from gymnasium import spaces
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import mss
from PIL import Image
import random

print()
JASS_WRITE_DIR =  r"C:\Users\Jamil\Documents\Warcraft III\CustomMapData"

def write_to_jass(filename, message):

        content = f"""function PreloadFiles takes nothing returns nothing

        call Preload( "" )
call BlzSetAbilityTooltip(1097690227,"-{str(message)}", 0)
//" )
        call Preload( "" )
endfunction
function a takes nothing returns nothing
//" )
        call PreloadEnd( 0.0 )

endfunction
        """
        try :
            with open(JASS_WRITE_DIR +'\\'+ filename, 'w') as file:
                file.write(content)

            print("Content written to " + filename )

        except :
                print("Error writing file to Jass retrying ...")
                write_to_jass(filename, message)


def extract_message_from_jass(path):
 
        with open(path, 'r') as file:
            lines = file.readlines()
            while lines == []:
                lines = file.readlines()
                time.sleep(0.1)
                 

            for line in lines:
                if 'BlzSetAbilityTooltip' in line:
                    start = line.find('-') + 1
                    end = line.find('"', start)
                    message = line[start:end]
        return message

GAME_PATH = r"C:\Warcraft 3\Warcraft III 1.31.1\x86_64\Warcraft III.exe"
WORKING_DIR = r"C:\Warcraft 3\Warcraft III 1.31.1\x86_64"
MAP_PATH = r"D:\Warcraft III modding\Warcraft III Move Bot\MainMap.w3x"
CUSTOM_MAPDATA_PATH = r"C:\Users\Jamil\Documents\Warcraft III\CustomMapData"
WGC_PATH = "C:/Warcraft 3/Warcraft III 1.31.1/map-wgc-test/MainMap-playtest.wgc"
APP_TITLE = "Warcraft III"

# scaler = MinMaxScaler(feature_range=(0, 1))
# ACTION_SPACE = np.array([[-1200, -3360], [1028, 2904]], dtype=np.float32)
# ACTION_SPACE = np.random.uniform(low=ACTION_SPACE[0], high=ACTION_SPACE[1], size=(10000, 2))


process_list = {}
ENV_SERIAL_NUM = 0

class Process():
     
     def __init__(self ,p_id , window_handle , process):
          
          self.process_id = p_id
          self.window_handle = window_handle
          self.process = process

def init_process(env_serial_num):
        
        proc = subprocess.Popen(
            [GAME_PATH, "-loadfile", WGC_PATH, "-nowfpause"],
            cwd=WORKING_DIR,
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )

        hwnd = 0
        while hwnd == 0:

            window_list = []
            win32gui.EnumWindows(lambda hwnd, window_list: window_list.append(hwnd) if win32gui.GetWindowText(hwnd) == APP_TITLE else None, window_list)
            for hdle in window_list :
                print()
                if win32process.GetWindowThreadProcessId(hdle)[1] == proc.pid:
                    hwnd = hdle
            time.sleep(0.1)
        

        process = Process(proc.pid, hwnd , proc)
        process_list[env_serial_num] = process

var = 0
def capture_window_screenshot(window_title = "Warcraft III" , hwnd =0):
    global var

    crop_size = 200
    try:
        # Find the window by title
        
        if not hwnd:
            raise Exception(f"Window with title '{window_title}' not found.")

        # Get the window's position and size
        rect = win32gui.GetWindowRect(hwnd)
        left, top, right, bottom = rect
        width = right - left
        height = bottom - top

        # Capture the screenshot of the specific window
        with mss.mss() as sct:
            monitor = {"top": top, "left": left, "width": width, "height": height}
            screenshot = sct.grab(monitor)

            # Save the screenshot
            output = "Observations\window_screenshot" +str(var) + ".png"
            var+=1
            # print(f"Screenshot saved to {output}")
            print(var)
            img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
            crop_area = (0, img.height-crop_size, crop_size, img.height)  # Adjust these values as needed
            
            cropped_img = img.crop(crop_area)
            cropped_img = cropped_img.resize((32, 32))

            return np.array(cropped_img)

    except Exception as e:
        print(f"Failed to capture screenshot: {e}")
        return None

     
class WarAMBOT(gym.Env):
    metadata = {"render.modes": ["human"]}

    def get_variable_value(self , path):

        variable = extract_message_from_jass(path)

        while variable is None :
            variable = extract_message_from_jass(path)

        return variable
    
    



    def write_env_number(self):

        global ENV_SERIAL_NUM

        access_permission = self.get_variable_value(CUSTOM_MAPDATA_PATH +"\\" + "env_number_access.txt")
        if self.env_num == 1 :
             access_permission = 1

        while int(access_permission) == 0:
            access_permission = self.get_variable_value(CUSTOM_MAPDATA_PATH +"\\" + "env_number_access.txt")
            time.sleep(0.1)       

        write_to_jass("env_number.txt" , str(ENV_SERIAL_NUM))
        write_to_jass("env_number_access.txt" , 0)

    def __init__(self) -> None:

        global  ENV_SERIAL_NUM
        ENV_SERIAL_NUM += 1
        self.env_num = ENV_SERIAL_NUM
        super(WarAMBOT , self).__init__()

        

        self.total_reward = 0 

        self.map_boundary_X = [-1100,1028]
        self.map_boundary_Y = [-3360,2905]

        self.observation_space = spaces.Box(low=0,
                                            high=255,
                                            shape=(128, 128, 128),
                                            dtype=np.uint8)

        
        self.action_space =     spaces.Box(
                                low=np.array([self.map_boundary_X[0], self.map_boundary_Y[0]]),
                                high=np.array([self.map_boundary_X[1], self.map_boundary_Y[1]]),
                                dtype=np.float32)

        self.action_path_x = r"C:\Users\Jamil\Documents\Warcraft III\CustomMapData\actionx_"+ str(self.env_num) + ".txt"
        self.action_path_y = r"C:\Users\Jamil\Documents\Warcraft III\CustomMapData\actiony_"+ str(self.env_num) + ".txt"
        self.ACTION_QUEUE_PATH = "action_queue_" + str(self.env_num) + ".txt"
        self.POSITIVE_REWARD_PATH = JASS_WRITE_DIR+"\\" +"negetive_reward_"+ str(self.env_num) + ".txt"
        self.NEGETIVE_REWARD_PATH = JASS_WRITE_DIR+"\\" +"positive_reward_" + str(self.env_num) + ".txt"
        self.DONE_PATH = JASS_WRITE_DIR+"\\" +"done_" + str(self.env_num) + ".txt"
        self.LOADING_PATH= JASS_WRITE_DIR+"\\" +"loading_finished_" + str(self.env_num) + ".txt"

   
        init_process(ENV_SERIAL_NUM)
       
        self.write_env_number()
        self.time_track = time.time()
        self.episode_duration = 30

        # Add if game loded or not check here________________ WAITING FOR THE INSTANCE TO LOAD __________________________________________

    def make_env(self):
        
        env = WarAMBOT()
        #Monitor(env ,"logs/")
        return Monitor(env)
    
    def close(self):
        process_list[ENV_SERIAL_NUM].process.terminate()
        print("Enviroment was Shut down")

            

    def send_action(self, action):
        write_to_jass("x_"+ str(self.env_num) + ".txt" , action[0])
        write_to_jass("y_"+ str(self.env_num) + ".txt" , action[1])
        
    def calculate_reward(self):

        positive_reward = extract_message_from_jass(self.POSITIVE_REWARD_PATH)
        negetive_reward = extract_message_from_jass(self.NEGETIVE_REWARD_PATH)

        return int(positive_reward) + int(negetive_reward)

            
    def reset(self , seed = 0):
        self.close()
        print("Reset Game State Initilized")
        init_process(self.env_num)
        time.sleep(3)
        observation = capture_window_screenshot( hwnd = process_list[self.env_num].window_handle)
        

        return observation , {}

    def step(self , action):
        
        #PLEASE DONT FORGET TO REMOVE THIS RANDOM ACTION GENERATION PLEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAASE OMG 
        action = [  self.action_space.sample() ]
        print()
        done = 0
        #time.sleep(3)
        print("Action: " , action)
        self.send_action(action)
        write_to_jass(self.ACTION_QUEUE_PATH , "1")
        keyboard_nofocus.press_key("left_arrow" , process_list[ENV_SERIAL_NUM].window_handle ,duration=0.1 , state="down")
        keyboard_nofocus.press_key("left_arrow" , process_list[ENV_SERIAL_NUM].window_handle ,duration=0.1 , state="up")
        
        action_queue_is_full = True
        temp = 1
        start_time = time.time()
        while action_queue_is_full:
            
            if int(temp) == 0 :
                    action_queue_is_full = False

            done =   extract_message_from_jass(self.DONE_PATH)
            temp =   extract_message_from_jass(JASS_WRITE_DIR +"\\"+ self.ACTION_QUEUE_PATH)

            if int(done)== 1:
                write_to_jass(self.ACTION_QUEUE_PATH , "1")
                loaded = False 

                while not loaded :
                    loaded = bool(extract_message_from_jass(self.LOADING_PATH))
                    print("Loading map ...")
                    time.sleep(0.1)

                keyboard_nofocus.press_key("left_arrow" , process_list[ENV_SERIAL_NUM].window_handle ,duration=0.1 , state="down")
                keyboard_nofocus.press_key("left_arrow" , process_list[ENV_SERIAL_NUM].window_handle ,duration=0.1 , state="up")
                action_queue_is_full = True


             


                break

            # if time.time() - start_time >= 6:
            #     time.sleep(6)
            #     keyboard_nofocus.press_key("left_arrow" , self.process.window_handle ,duration=0.1 , state="down")
            #     keyboard_nofocus.press_key("left_arrow" , self.process.window_handle ,duration=0.1 , state="up")
            #     action_queue_is_full = False
            #     break
                

        reward = self.calculate_reward()
        done =   extract_message_from_jass(self.DONE_PATH)

        if int(done) == 0 :
             done = False
        else:
             done = True
             write_to_jass("done_" + str(self.env_num) + ".txt", 0)
        

        print("Total step reward : " , str(reward) , " Enviroment: " , self.env_num)
        self.total_reward += int(reward)
        print("Total reward :" , self.total_reward , " Enviroment: " , self.env_num)

        observation = action
        observation = []
        info = {}
        truncated = False
        print()
        return observation  ,reward, done, truncated, info

 
        
            