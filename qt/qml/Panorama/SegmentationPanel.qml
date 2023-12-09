import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Controls.Material 2.15
import QtQuick.Layouts 1.15
import QtQuick.Window 2.15
import QtGraphicalEffects 1.0
import "../Custom"
import Backend 1.0

Pane {
    function init() {
    }

    function update_image_view(paths) {
        image_model.clear();
        paths.forEach((path) => image_model.append({
            "path": path
        }));
    }

    width: 1280
    height: 720
    padding: 10
    objectName: 'segmentation_panel'

    ColumnLayout {
        anchors.fill: parent

        ToolBar {
            ToolButton {
                text: qsTr('부위 인식')
                icon: '\uea40'
                onReleased: con.command('segment')
                ToolTip.visible: hovered
                ToolTip.delay: 500
                ToolTip.text: qsTr('정합된 실화상의 자동 부위 인식 실행')
            }

        }

        Pane {
            Material.elevation: 2
            Layout.fillHeight: true
            Layout.fillWidth: true
            padding: 0

            FigureCanvas {
                id: plot

                anchors.fill: parent
                objectName: 'segmentation_plot'
                dpi_ratio: Screen.devicePixelRatio
            }

        }

        Pane {
            Material.elevation: 2
            Layout.fillWidth: true
            Layout.preferredHeight: 200
            padding: 5

            ListView {
                id: image_view

                anchors.fill: parent
                clip: true
                orientation: ListView.Horizontal

                WheelHandler {
                    onWheel: {
                        if (event.angleDelta.y < 0)
                            _sb.increase();
                        else
                            _sb.decrease();
                    }
                }

                ScrollBar.horizontal: ScrollBar {
                    id: _sb

                    policy: ScrollBar.AsNeeded
                }

                model: ListModel {
                    id: image_model
                }

                delegate: Pane {
                    Material.elevation: 0
                    height: image_view.height - 20
                    width: height * 4 / 3 + 10

                    Image {
                        id: _image

                        source: path
                        width: parent.width
                        fillMode: Image.PreserveAspectFit
                    }

                    BrightnessContrast {
                        id: _bc

                        anchors.fill: _image
                        source: _image
                        brightness: 0
                    }

                    MouseArea {
                        anchors.fill: parent
                        hoverEnabled: true
                        onReleased: con.seg_plot(path)
                        onEntered: _bc.brightness = -0.25
                        onExited: _bc.brightness = 0
                    }

                }

            }

        }

    }

}
