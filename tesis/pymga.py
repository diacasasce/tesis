import sys
from PyQt4.QtGui import *
app = QApplication(sys.argv)
label = QLabel()
pixmap = QPixmap(sys.argv[0])
label.setPixmap(pixmap)
label.show()
sys.exit(app.exec_())