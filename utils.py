def get_device_list_and_log(selection, num_devices):
    if selection == 0:
        device_list = [0, 1, 16]
        img_dir = "./output/num_3/"
        log_filename = "./output/num_3.txt"
    elif selection == 1:
        device_list = list(range(num_devices))
        img_dir = "./output/num_17/"
        log_filename = "./output/num_17.txt"
    elif selection == 2:
        device_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        img_dir = "./output/num_8_IIP3_1/"
        log_filename = "./output/num_8_IIP3_1.txt"
    elif selection == 3:
        device_list = [0, 9, 10, 11, 12, 13, 14, 15, 16]
        img_dir = "./output/num_8_IIP3_2/"
        log_filename = "./output/num_8_IIP3_2.txt"
    elif selection == 4:
        device_list = [0, 1, 2, 3, 4, 9, 10, 11, 12]
        img_dir = "./output/num_8_IQ_imbal_1/"
        log_filename = "./output/num_8_IQ_imbal_1.txt"
    elif selection == 5:
        device_list = [0, 5, 6, 7, 8, 13, 14, 15, 16]
        img_dir = "./output/num_8_IQ_imbal_2/"
        log_filename = "./output/num_8_IQ_imbal_2.txt"
    elif selection == 6:
        device_list = [0, 1, 2, 5, 6, 9, 10, 13, 14]
        img_dir = "./output/num_8_DC_1/"
        log_filename = "./output/num_8_DC_1.txt"
    elif selection == 7:
        device_list = [0, 3, 4, 7, 8, 11, 12, 15, 16]
        img_dir = "./output/num_8_DC_2/"
        log_filename = "./output/num_8_DC_2.txt"
    elif selection == 8:
        device_list = [0, 1, 3, 5, 7, 9, 11, 13, 15]
        img_dir = "./output/num_8_phase_O_1/"
        log_filename = "./output/num_8_phase_O_1.txt"
    elif selection == 9:
        device_list = [0, 2, 4, 6, 8, 10, 12, 14, 16]
        img_dir = "./output/num_8_phase_O_2/"
        log_filename = "./output/num_8_phase_O_2.txt"
    else:
        device_list = list(range(num_devices))
        img_dir = "./output/num_17/"
        log_filename = "./output/num_17.txt"
    return device_list, img_dir, log_filename
