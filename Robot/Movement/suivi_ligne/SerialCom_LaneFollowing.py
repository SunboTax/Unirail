# On importe les bibliotheques / fonctions utiles
import serial
from time import sleep
import random as rd
from os import system

end_key = "UniRAIL"

#On cree une classe symbolisant le MegaPi (question de facilite)
class MegaPi(serial.Serial) :
    
    #Le constructeur initialise la liaison serie avec le MegaPi
    def __init__(self, desc: bool = False, serial_port: int = 0, baud: int = 115200) :
        super().__init__(port="/dev/serial"+str(serial_port), baudrate=baud, timeout=1)
        if desc :
            print("--------------------------------------------------")
            print("New MegaPi linked to the Raspberry!")
            print("  * serial port used: " + self.port)
            print("  * baudrate: " + str(self.baudrate))
            print("  * timeout: " + str(self.timeout) + "s")
            print("--------------------------------------------------\n")
    
    #La methode pour faire fonctionner un moteur
    def sendThetaEpsilonU(self, theta: int, epsilon : int, u : float) :
        program = "T" + "{:+d}".format(theta) + "E" + "{:+d}".format(epsilon) + "U" + "{:+d}".format(int(u*100)) + str(end_key)
        self.write(program.encode())
        return_msg = self.read(size=3+len(program))
        if return_msg.decode() != "OK_" + program :
            print("OK_"+program)
            print(return_msg.decode())
            print("Connection with MegaPi has somehow failed. Try again!")

    def endCom(self) :
        self.sendThetaEpsilonU(0, 0, 0)
        sleep(0.1)
        self.close()
        
    def startCom(self) :
        self.open()