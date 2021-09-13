import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Controls.Material 2.15
import QtQuick.Dialogs 1.3
import QtQuick.Layouts 1.15
import QtQuick.Window 2.15

import 'Custom'
import Backend 1.0


Pane {
    width : 1280
    height : 720
    padding : 10
    objectName : 'panorama_panel'

    ColumnLayout {
        anchors.fill : parent

        ToolBar {
            RowLayout {
                spacing : 10

                ToolButton {
                    text : qsTr('파노라마 생성')
                    icon : '\ue40b'
                    onReleased : {
                        app.pb_state(true);
                        con.command('panorama');
                    }
                }
                ToolButton {
                    text : qsTr('자동 보정')
                    icon : '\ue663'
                    onReleased : {
                        app.pb_state(true);
                        con.command('correct');
                    }
                }

                ToolSeparator {}

                ToolButton {
                    text : qsTr('저장')
                    icon : '\ue161'
                    onReleased : con.pano_save_manual_correction(_roll.value, _pitch.value, _yaw.value)
                }
                ToolButton {
                    text : qsTr('취소')
                    icon : '\ue14a'
                    onReleased : reset()
                }
            }
        }

        Pane {
            Material.elevation : 2
            Layout.fillHeight : true
            Layout.fillWidth : true
            padding : 0

            FigureCanvas {
                id : plot
                anchors.fill : parent
                objectName : 'panorama_plot'
                Layout.fillHeight : true
                Layout.fillWidth : true
                dpi_ratio : Screen.devicePixelRatio
            }
        }

        RowLayout {
            Pane {
                Material.elevation : 2
                Layout.fillWidth : true
                Layout.preferredHeight : 140

                RowLayout {
                    anchors.fill : parent
                    Layout.alignment : Qt.AlignVCenter

                    ColumnLayout {

                        RowLayout {
                            Label {
                                text : '\ue028'
                                font.family : 'Material Icons'
                                font.pointSize : 18
                                Layout.preferredWidth : 30
                            }
                            Label {
                                text : qsTr('Roll')
                                Layout.preferredWidth : 50
                            }
                            BiSlider {
                                id : _roll
                                Layout.fillWidth : true

                                from : -90
                                to : 90

                                onValueChanged : rotate()
                            }
                        }
                        RowLayout {
                            Label {
                                text : '\ue0c3'
                                font.family : 'Material Icons'
                                font.pointSize : 18
                                Layout.preferredWidth : 30
                            }
                            Label {
                                text : qsTr('Pitch')
                                Layout.preferredWidth : 50
                            }
                            BiSlider {
                                id : _pitch
                                Layout.fillWidth : true

                                from : -60
                                to : 60

                                onValueChanged : rotate()
                            }
                        }
                        RowLayout {
                            Label {
                                text : '\ue8d4'
                                font.family : 'Material Icons'
                                font.pointSize : 18
                                Layout.preferredWidth : 30
                            }
                            Label {
                                text : qsTr('Yaw')
                                Layout.preferredWidth : 50
                            }
                            BiSlider {
                                id : _yaw
                                Layout.fillWidth : true

                                from : -60
                                to : 60

                                onValueChanged : rotate()
                            }
                        }
                    }
                }
            }
            Pane {
                Material.elevation : 2
                Layout.preferredHeight : 140

                ColumnLayout {
                    anchors.fill : parent

                    RowLayout {
                        Label {
                            text : qsTr('해상도')
                        }
                        SpinBox {
                            id : _resolution
                            from : 200
                            to : 2000
                            value : 1000
                            stepSize : 100
                        }
                    }

                    RowLayout {
                        Label {
                            text : qsTr('그리드')
                        }
                        CheckBox {
                            checkState : Qt.Checked
                            onCheckStateChanged : con.pano_set_grid(checkState == Qt.Checked)
                        }
                    }
                }
            }
        }
    }

    function init() {
        con.pano_plot()
    }

    function set_image(url) {
        _image.source = url
    }

    function rotate() {
        con.pano_rotate(_roll.value, _pitch.value, _yaw.value, _resolution.value)
    }

    function reset() {
        _roll.value = 0
        _pitch.value = 0
        _yaw.value = 0
    }
}
