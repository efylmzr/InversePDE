from utils import redistance
import mitsuba as mi 
mi.set_variant("cuda_ad_rgb")
def main():
    resolution = 1024
    redistance("motorbike-engine", resolution, method = 2)
    #print("m is redistanced.")
    #redistance("engine", resolution, rotate = True)
    #print("Engine is redistanced.")

if __name__ == "__main__":
    main()

