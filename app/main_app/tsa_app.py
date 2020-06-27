import sys
from PyQt5.QtWidgets import QApplication, QFileDialog
from PyQt5 import QtWidgets, QtCore
from app.main_app.tsa_ui import Ui_MainWindow
import pandas as pd
import matplotlib

matplotlib.use('Qt5Agg')


class PandasModel(QtCore.QAbstractTableModel):
    """
    Class to populate a table view with a pandas dataframe
    """

    def __init__(self, data, parent=None):
        QtCore.QAbstractTableModel.__init__(self, parent)
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parent=None):
        return self._data.shape[1]

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if index.isValid():
            if role == QtCore.Qt.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, section: int, orientation: QtCore.Qt.Orientation, role: int = ...):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return self._data.columns[section]
        if orientation == QtCore.Qt.Vertical and role == QtCore.Qt.DisplayRole:
            return self._data.index[section]
        return None


class TSApp(QtWidgets.QMainWindow, Ui_MainWindow):
    df = None
    type_conversion = {'int': int, 'float': float, 'bool': bool, 'string': str}
    data_filename = None

    def __init__(self):
        super(TSApp, self).__init__()
        self.setupUi(self)
        self.show()

        # Open File
        self.actionOpen.triggered.connect(self.open_filename_dialog)
        self.actionSave.triggered.connect(self.save_df_to_file)

        # Index Section:
        self.setIndexButton.clicked.connect(self.set_index)
        self.resetIndexButton.clicked.connect(self.reset_index)
        self.resetIndexDropButton.clicked.connect(self.reset_index_drop)

        # Drop Columns Section:
        self.dropColumnButton.clicked.connect(self.drop_column)

        # Type Modifier:
        self.modifyTypeColumnSelector.activated.connect(self.get_current_column_type)
        self.ModifyTypeConfirm.clicked.connect(self.modify_column_type)

        # Sorting
        self.sortColumnSelector.activated.connect(self.sort_dataframe)
        self.sortByIndex.clicked.connect(self.sort_dataframe_by_index)
        self.ascendingCheck.clicked.connect(self.sort_dataframe)

        # Reset
        self.resetDataFrame.clicked.connect(self.load_dataframe)

    def open_filename_dialog(self):
        qfile_dialoge = QFileDialog()
        filename, _ = QFileDialog.getOpenFileName(qfile_dialoge, "Open File")
        print(filename)
        self.data_filename = filename
        self.load_dataframe()

    def load_dataframe(self):
        file_extension = self.data_filename.split('.')[1]
        if file_extension == 'csv':
            self.df = pd.read_csv(self.data_filename)
        elif file_extension == 'parquet':
            self.df = pd.read_parquet(self.data_filename)
        self.update_dataframe()

    def update_dataframe(self):
        self.populate_initialization_tab()
        self.populate_dataframe_view()
        self.populate_column_selectors()

    def populate_initialization_tab(self):
        self.nRowsLabel.setText(str(self.df.shape[0]))
        self.nColumnsLabel.setText(str(self.df.shape[1]))
        self.sizeMBLabel.setText(str(round(self.df.memory_usage(deep=True).sum() / 1024 ** 2, 3)))

    def populate_dataframe_view(self):
        model = PandasModel(self.df)
        self.dataFrameTableView.setModel(model)

    def populate_column_selectors(self):
        dataframe_columns = self.df.columns

        for selector in [self.modifyTypeColumnSelector, self.setIndexColumnSelector,
                         self.sortColumnSelector, self.dropColumnSelector]:
            selector.clear()
            selector.addItems(dataframe_columns)

        self.modifyTypeTypeSelector.clear()
        self.modifyTypeTypeSelector.addItems(['float', 'int', 'bool', 'string', 'datetime'])

    def set_index(self):
        column_index = self.setIndexColumnSelector.currentText()

        if isinstance(self.df.index, pd.RangeIndex):
            self.reset_index_drop()
        else:
            self.reset_index()

        self.df = self.df.set_index(column_index)
        self.update_dataframe()

    def reset_index(self):
        self.df = self.df.reset_index()
        self.update_dataframe()

    def reset_index_drop(self):
        self.df = self.df.reset_index(drop=True)
        self.update_dataframe()

    def drop_column(self):
        column = self.dropColumnSelector.currentText()
        self.df = self.df.drop(column, axis=1)
        self.update_dataframe()

    def modify_column_type(self):
        column = self.modifyTypeColumnSelector.currentText()
        to_type = self.modifyTypeTypeSelector.currentText()

        try:
            if to_type == 'datetime':
                self.df[column] = pd.to_datetime(self.df[column])
            else:
                self.df[column] = self.df[column].astype(self.type_conversion[to_type])
        except Exception as ex:
            print(ex)
        self.update_dataframe()
        self.get_current_column_type()

    def get_current_column_type(self):
        column = self.modifyTypeColumnSelector.currentText()
        column_type = self.df[column].dtype
        self.currentColumnTypeLabel.setText(str(column_type))

    def sort_dataframe(self):
        self.df = self.df.sort_values(by=self.sortColumnSelector.currentText(),
                                      ascending=self.ascendingCheck.isChecked())
        self.populate_dataframe_view()

    def sort_dataframe_by_index(self):
        self.df = self.df.sort_index()
        self.populate_dataframe_view()

    def save_df_to_file(self):
        qfile_dialoge = QFileDialog()
        filename, _ = QFileDialog.getSaveFileName(qfile_dialoge, 'Save File')
        file_extension = filename.split('.')[1]

        if file_extension == 'csv':
            self.df.to_csv(filename)
        elif file_extension == 'parquet':
            self.df.to_parquet(filename)

        print(filename)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = TSApp()
    sys.exit(app.exec_())
