import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Controls.Material 2.15
import QtQuick.Dialogs 1.3
import QtQuick.Layouts 1.15
import QtQuick.Window 2.15

import 'Custom'
import 'OptionPopup'
import Backend 1.0


Pane {
    width : 1280
    height : 720
    padding : 10
    objectName : 'panorama_panel'

    // TODO cmap range

    property bool correction_plot : false

    PanoramaOption {
        id : _panorama_option
    }
    CorrectionOption {
        id : _correction_option
    }

    ColumnLayout {
        anchors.fill : parent

        ToolBar {
            RowLayout {
                spacing : 0

                RowLayout {
                    visible : !correction_plot

                    ToolButton {
                        text : qsTr('파노라마 생성')
                        icon : '\ue40b'
                        onReleased : {
                            app.pb_state(true);
                            con.command('panorama');
                            _ir.checked = true;
                        }

                        ToolTip.visible : hovered
                        ToolTip.delay : 500
                        ToolTip.text : qsTr('열화상 정합을 통해 파노라마 생성')
                    }

                    ToolSeparator {}
                }


                RowLayout {
                    ToolRadioButton {
                        id : _ir
                        text : '열화상'
                        checked : true

                        onReleased : pano_plot()
                    }
                    ToolRadioButton {
                        id : _vis
                        text : '실화상'

                        onReleased : pano_plot()
                    }
                    ToolRadioButton {
                        id : _seg
                        text : '부위 인식'

                        onReleased : pano_plot()
                    }
                }

                ToolSeparator {}

                RowLayout {
                    visible : correction_plot

                    ToolButton {
                        text : qsTr('자동 보정')
                        icon : '\ue663'
                        onReleased : {
                            app.pb_state(true);
                            con.command('correct');
                            _ir.checked = true;
                        }

                        ToolTip.visible : hovered
                        ToolTip.delay : 500
                        ToolTip.text : qsTr('시점 왜곡 자동 보정')
                    }

                    ToolSeparator {}

                    ToolButton {
                        id : _manual
                        text : qsTr('수동 보정')
                        icon : '\ue41e'

                        down : true
                        onReleased : {
                            down = true;
                            _crop.down = false;
                        }

                        ToolTip.visible : hovered
                        ToolTip.delay : 500
                        ToolTip.text : qsTr('사용자가 지정한 각도에 따라 시점 왜곡을 수동으로 보정')
                    }
                    ToolButton {
                        id : _crop
                        text : qsTr('자르기')
                        icon : '\ue3be'

                        onReleased : {
                            down = true;
                            _manual.down = false;
                        }
                        onDownChanged : con.pano_crop_mode(down)

                        ToolTip.visible : hovered
                        ToolTip.delay : 500
                        ToolTip.text : qsTr('시점 왜곡이 보정된 영상의 저장 영역을 마우스 드래그를 통해 지정')
                    }

                    ToolSeparator {}

                    ToolButton {
                        text : qsTr('저장')
                        icon : '\ue161'
                        onReleased : con.pano_save_manual_correction(_roll.value, _pitch.value, _yaw.value)

                        ToolTip.visible : hovered
                        ToolTip.delay : 500
                        ToolTip.text : qsTr('수동 시점 왜곡 보정·영역 지정 결과를 저장')
                    }
                    ToolButton {
                        text : qsTr('취소')
                        icon : '\ue14a'
                        onReleased : {
                            if (_manual.down) {
                                reset()
                            } else {
                                con.pano_home()
                            }
                        }

                        ToolTip.visible : hovered
                        ToolTip.delay : 500
                        ToolTip.text : qsTr('수동 시점 왜곡 보정·영역 지정 취소')
                    }

                    ToolSeparator {}
                }

                ToolButton {
                    text : qsTr('설정')
                    icon : '\ue8b8'

                    onReleased : {
                        if (correction_plot) {
                            _correction_option.open()
                        } else {
                            _panorama_option.open()
                        }
                    }

                    ToolTip.visible : hovered
                    ToolTip.delay : 500
                    ToolTip.text : (correction_plot ? '시점 왜곡 보정 설정' : '파노라마 생성 설정')
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
            visible : correction_plot

            Pane {
                Material.elevation : 2
                Layout.fillWidth : true
                Layout.preferredHeight : 140

                ToolTip.visible : hovered
                ToolTip.delay : 500
                ToolTip.timeout : 2000
                ToolTip.text : qsTr('시점 왜곡을 보정하기 위한 촬영 각도 변경')

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
                                enabled : _manual.down

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
                                enabled : _manual.down

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
                                enabled : _manual.down

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
                    spacing : 0

                    Pane {
                        Layout.preferredHeight : 50

                        ToolTip.visible : hovered
                        ToolTip.delay : 500
                        ToolTip.text : qsTr('수동 시점 보정 결과의 시각화를 위한 영상의 해상도 (저장 해상도와 다름)')

                        RowLayout {
                            anchors.fill : parent

                            Label {
                                text : qsTr('해상도')
                            }
                            SpinBox {
                                id : _resolution
                                Layout.alignment : Qt.AlignRight | Qt.AlignVCenter
                                from : 200
                                to : 2000
                                value : 1000
                                stepSize : 100
                            }
                        }
                    }

                    RowLayout {
                        Pane {
                            Layout.preferredHeight : 50

                            RowLayout {
                                Label {
                                    text : qsTr('그리드')
                                }
                                CheckBox {
                                    checkState : Qt.Checked
                                    onCheckStateChanged : con.pano_set_grid(checkState == Qt.Checked)
                                }

                                Rectangle {
                                    Layout.preferredWidth : 10
                                }
                            }
                        }

                        Pane {
                            Layout.fillWidth : true
                            Layout.preferredHeight : 50

                            ToolTip.visible : hovered
                            ToolTip.delay : 500
                            ToolTip.text : qsTr('열화상 카메라의 시야각')

                            RowLayout {
                                Label {
                                    text : qsTr('시야각')
                                }
                                Rectangle {
                                    Layout.preferredWidth : 5
                                }
                                TextField {
                                    text : '42'
                                    Layout.preferredWidth : 40

                                    validator : DoubleValidator {}
                                    onTextChanged : {
                                        con.pano_set_viewing_angle(text)
                                    }
                                }
                                Label {
                                    text : qsTr('º')
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    function init() {
        con.pano_plot((correction_plot ? 'COR' : 'PANO'), 'IR')
    }

    function pano_plot() {
        let sp = _ir.checked ? 'IR' : (_vis.checked ? 'VIS' : 'SEG')
        let d = correction_plot ? 'COR' : 'PANO'
        con.pano_plot(d, sp)
        reset()
    }

    function rotate() {
        con.pano_rotate(_roll.value, _pitch.value, _yaw.value, _resolution.value)
    }

    function reset() {
        _roll.value = 0
        _pitch.value = 0
        _yaw.value = 0
    }

    function update_config(config) {
        _panorama_option.update_config(config)
        _correction_option.update_config(config)
    }
}
