import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Controls.Material 2.15
import QtQuick.Dialogs 1.3
import QtQuick.Layouts 1.15
import QtQuick.Window 2.15

import Backend 1.0


Pane {
    width : 1280
    height : 720
    padding : 10
    objectName : 'segmentation_panel'

    ColumnLayout {
        anchors.fill : parent

        RowLayout {
            Button {
                text : qsTr('부위 인식')
                onReleased : con.command('segment')
            }
        }

        Pane {
            Material.elevation : 2
            Layout.fillHeight : true
            Layout.fillWidth : true

            ColumnLayout {
                anchors.fill : parent

                FigureCanvas {
                    id : plot
                    objectName : 'segmentation_plot'
                    Layout.fillHeight : true
                    Layout.fillWidth : true
                    dpi_ratio : Screen.devicePixelRatio
                }
            }
        }

        Pane {
            Material.elevation : 2
            Layout.fillWidth : true
            Layout.preferredHeight : 200

            ListView {
                id : image_view

                anchors.fill : parent
                clip : true
                orientation : ListView.Horizontal

                ScrollBar.horizontal : ScrollBar {
                    policy : ScrollBar.AsNeeded
                }

                model : ListModel {
                    id : image_model
                }

                delegate : Pane {
                    Material.elevation : 0
                    height : image_view.height - 10
                    width : height * 4 / 3 + 10

                    Image {
                        source : path
                        width : parent.width
                        fillMode : Image.PreserveAspectFit
                    }

                    MouseArea {
                        anchors.fill : parent
                        hoverEnabled : true

                        onReleased : con.seg_plot(path)
                    }
                }
            }
        }
    }

    function update_image_view(paths) {
        image_model.clear()
        paths.forEach(path => image_model.append({'path': path}))
    }
}
