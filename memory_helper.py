import psutil
import win32api, win32con, win32process, win32
from ctypes import create_string_buffer, c_ulonglong, c_ulong, byref, windll



def get_pid(process_name):
    for proc in psutil.process_iter():
        try:
            if proc.name() == process_name:
                return proc.pid
        except psutil.AccessDenied:
            print("Permission error or access denied on process")

def calculate_mono_address(pid):
    phandle = win32api.OpenProcess(win32con.MAXIMUM_ALLOWED, 0, pid)
    mono_addr = None

    for module in win32process.EnumProcessModules(phandle):
        path = win32process.GetModuleFileNameEx(phandle, module)
        mod_name = path.split('\\')[-1]
        if mod_name == 'mono.dll':
            #print(mod_name, hex(module))
            mono_addr = module

    return mono_addr, phandle


def calculate_final_address(pid, offsets):
    mono_addr, phandle = calculate_mono_address(pid)

    offset = 0x00266618
    mono_addr += offset

    #print("mono + offset0", hex(mono_addr))



    bufferSize = 8
    buffer = create_string_buffer(bufferSize)
    bytesRead = c_ulong(0)

    current_addr = mono_addr

    for of in offsets:
        #read contents of current_addr into buffer
        windll.kernel32.ReadProcessMemory(phandle.handle, c_ulonglong(current_addr), buffer, bufferSize, byref(bytesRead))

        pvalues = []

        #create hex value from byte pieces
        for i in range(0, bufferSize):
            pvalues.append(buffer[i].hex())      
        pvalue = '0x' + ''.join([format(int(c, 16), '02X') for c in reversed(pvalues)])
        
        #print(f"adding {hex(of)} to {hex(int(pvalue, 16))}")

        # add offset to the pointer
        pvalue = (hex(int(pvalue, 16)+of))  
        #print("result:",hex((int(pvalue, 16))))

        current_addr = int(pvalue, 16)

            
    return current_addr, phandle


def read_value_from_memory(score_address, phandle):

    bufferSize = 4
    buffer = create_string_buffer(bufferSize)
    bytesRead = c_ulong(0)

    # read the score from the memory location
    windll.kernel32.ReadProcessMemory(phandle.handle, c_ulonglong(score_address), buffer, bufferSize, byref(bytesRead))

    pvalues = []

    #create hex value from byte pieces
    for i in range(0, bufferSize):
        pvalues.append(buffer[i].hex())      
    pvalue = '0x' + ''.join([format(int(c, 16), '02X') for c in reversed(pvalues)])

    #print(hex(score_address), int(pvalue, 16))

    return int(pvalue, 16)
    

if __name__ == '__main__':
    pid = get_pid('superflight.exe')
    score_addr, phandle = calculate_final_address(pid, [0xA0, 0x2F0, 0x20, 0x9C])
    score = read_value_from_memory(score_addr, phandle)

    print(score)