import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Controls.Material 2.15
import QtQuick.Layouts 1.15
import QtQuick.Window 2.15
import "../Custom"
import "../Button" as Btn
import "OptionPopup" as Opt
import Backend 1.0

Pane {
    property bool correction_plot: false

    function init() {
        con.pano_plot((correction_plot ? 'COR' : 'PANO'), 'IR');
    }

    function pano_plot() {
        let sp = _ir.checked ? 'IR' : (_vis.checked ? 'VIS' : 'SEG');
        let d = correction_plot ? 'COR' : 'PANO';
        con.pano_plot(d, sp);
        reset();
    }

    function rotate() {
        con.pano_rotate(_roll.value, _pitch.value, _yaw.value, _resolution.value);
    }

    function reset() {
        _roll.value = 0;
        _pitch.value = 0;
        _yaw.value = 0;
    }

    function update_config(config) {
        _panorama_option.update_config(config);
        _correction_option.update_config(config);
    }

    width: 1280
    height: 720
    padding: 10
    objectName: 'panorama_panel'

    Opt.Panorama {
        id: _panorama_option
    }

    Opt.Correction {
        id: _correction_option
    }

    ColumnLayout {
        anchors.fill: parent

        ToolBar {
            RowLayout {
                spacing: 0

                RowLayout {
                    visible: !correction_plot

                    Btn.ToolButton {
                        text: qsTr('파노라마 생성')
                        icon: '\ue40b'
                        ToolTip.visible: hovered
                        ToolTip.delay: 500
                        ToolTip.text: qsTr('열화상 정합을 통해 파노라마 생성')
                        onReleased: {
                            app.pb_state(true);
                            con.command('panorama');
                            _ir.checked = true;
                        }
                    }

                }

                RowLayout {
                    visible: correction_plot

                    Btn.ToolButton {
                        text: qsTr('자동 보정')
                        icon: '\ue663'
                        ToolTip.visible: hovered
                        ToolTip.delay: 500
                        ToolTip.text: qsTr('시점 왜곡 자동 보정')
                        onReleased: {
                            app.pb_state(true);
                            con.command('correct');
                            _ir.checked = true;
                        }
                    }

                    Btn.ToolButton {
                        text: qsTr('저장')
                        icon: '\ue161'
                        onReleased: con.pano_save_manual_correction(_roll.value, _pitch.value, _yaw.value)
                        ToolTip.visible: hovered
                        ToolTip.delay: 500
                        ToolTip.text: qsTr('수동 시점 왜곡 보정·영역 지정 결과를 저장')
                    }

                }

                ToolSeparator {
                }

                Btn.OpenFolder {
                    onReleased: con.open_dir(correction_plot ? 'COR' : 'PANO')
                }

                ToolSeparator {
                }

                Btn.Navigation {
                    index: correction_plot ? 5 : 3
                }

                ToolSeparator {
                }

                Btn.Setting {
                    ToolTip.text: (correction_plot ? '시점 왜곡 보정 설정' : '파노라마 생성 설정')
                    onReleased: {
                        if (correction_plot)
                            _correction_option.open();
                        else
                            _panorama_option.open();
                    }
                }

                Btn.Help {
                    // TODO

                }

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
                objectName: 'panorama_plot'
                dpi_ratio: Screen.devicePixelRatio
            }

            Pane {
                // plot 상호작용 버튼
                Material.elevation: 1
                padding: 0
                leftPadding: 5
                anchors.left: parent.left
                anchors.top: parent.top

                RowLayout {
                    RowLayout {
                        visible: _expand.expanded

                        RowLayout {
                            // 왜곡보정 plot 상호작용 버튼
                            visible: correction_plot

                            Btn.MiniToolButton {
                                id: _manual

                                text: '수동 보정'
                                icon: '\ue41e'
                                down: true
                                ToolTip.text: '사용자가 지정한 각도에 따라 시점 왜곡을 수동으로 보정'
                                onReleased: {
                                    down = true;
                                    _crop.down = false;
                                }
                            }

                            Btn.MiniToolButton {
                                id: _crop

                                text: '자르기'
                                icon: '\ue3be'
                                onDownChanged: con.pano_crop_mode(down)
                                ToolTip.text: '시점 왜곡이 보정된 영상의 저장 영역을 마우스 드래그를 통해 지정'
                                onReleased: {
                                    down = true;
                                    _manual.down = false;
                                }
                            }

                            Btn.MiniToolButton {
                                text: '취소'
                                icon: '\ue14a'
                                ToolTip.text: '수동 시점 왜곡 보정·영역 지정 취소'
                                onReleased: {
                                    if (_manual.down)
                                        reset();
                                    else
                                        con.pano_home();
                                }
                            }

                            ToolSeparator {
                                leftPadding: 2
                                rightPadding: 2
                            }

                            Btn.MiniToolButton {
                                text: '그리드'
                                icon: '\ue3ec'
                                ToolTip.text: '그리드 표시 여부'
                                onReleased: {
                                    down = !down;
                                    con.pano_set_grid(down);
                                }
                            }

                            ToolSeparator {
                                leftPadding: 2
                                rightPadding: 2
                            }

                        }

                        RadioButton {
                            id: _ir

                            implicitHeight: 36
                            text: '열화상'
                            checked: true
                            onReleased: pano_plot()
                        }

                        RadioButton {
                            id: _vis

                            implicitHeight: 36
                            text: '실화상'
                            onReleased: pano_plot()
                        }

                        RadioButton {
                            id: _seg

                            implicitHeight: 36
                            text: '부위 인식'
                            onReleased: pano_plot()
                        }

                    }

                    Btn.Expand {
                        id: _expand

                        padding: 0
                    }

                }

            }

        }

        RowLayout {
            visible: correction_plot

            Pane {
                Material.elevation: 2
                Layout.fillWidth: true
                Layout.preferredHeight: 140
                ToolTip.visible: hovered
                ToolTip.delay: 500
                ToolTip.timeout: 2000
                ToolTip.text: qsTr('시점 왜곡을 보정하기 위한 촬영 각도 변경')

                RowLayout {
                    anchors.fill: parent
                    Layout.alignment: Qt.AlignVCenter

                    ColumnLayout {
                        RowLayout {
                            Label {
                                text: '\ue028'
                                font.family: 'Material Symbols Outlined'
                                font.pointSize: 18
                                Layout.preferredWidth: 30
                            }

                            Label {
                                text: qsTr('Roll')
                                Layout.preferredWidth: 50
                            }

                            BiSlider {
                                id: _roll

                                Layout.fillWidth: true
                                enabled: _manual.down
                                from: -90
                                to: 90
                                onValueChanged: rotate()
                            }

                        }

                        RowLayout {
                            Label {
                                text: '\ue0c3'
                                font.family: 'Material Symbols Outlined'
                                font.pointSize: 18
                                Layout.preferredWidth: 30
                            }

                            Label {
                                text: qsTr('Pitch')
                                Layout.preferredWidth: 50
                            }

                            BiSlider {
                                id: _pitch

                                Layout.fillWidth: true
                                enabled: _manual.down
                                from: -60
                                to: 60
                                onValueChanged: rotate()
                            }

                        }

                        RowLayout {
                            Label {
                                text: '\ue8d4'
                                font.family: 'Material Symbols Outlined'
                                font.pointSize: 18
                                Layout.preferredWidth: 30
                            }

                            Label {
                                text: qsTr('Yaw')
                                Layout.preferredWidth: 50
                            }

                            BiSlider {
                                id: _yaw

                                Layout.fillWidth: true
                                enabled: _manual.down
                                from: -60
                                to: 60
                                onValueChanged: rotate()
                            }

                        }

                    }

                }

            }

            Pane {
                Material.elevation: 2
                Layout.preferredHeight: 140

                ColumnLayout {
                    anchors.fill: parent
                    spacing: 0

                    Pane {
                        Layout.preferredHeight: 50
                        ToolTip.visible: hovered
                        ToolTip.delay: 500
                        ToolTip.text: qsTr('수동 시점 보정 결과의 시각화를 위한 영상의 해상도 (저장 해상도와 다름)')

                        RowLayout {
                            anchors.fill: parent

                            Label {
                                Layout.preferredWidth: 60
                                text: '해상도'
                            }

                            SpinBox {
                                id: _resolution

                                Layout.alignment: Qt.AlignRight | Qt.AlignVCenter
                                from: 200
                                to: 2000
                                value: 1000
                                stepSize: 100
                            }

                        }

                    }

                    Pane {
                        Layout.fillWidth: true
                        Layout.preferredHeight: 50
                        ToolTip.visible: hovered
                        ToolTip.delay: 500
                        ToolTip.text: qsTr('열화상 카메라의 시야각')

                        RowLayout {
                            Label {
                                Layout.preferredWidth: 60
                                text: '시야각 (º)'
                            }

                            SpinBox {
                                Layout.alignment: Qt.AlignRight | Qt.AlignVCenter
                                from: 20
                                to: 120
                                value: 42
                                stepSize: 2
                                onValueChanged: con.pano_set_viewing_angle(value)
                            }

                        }

                    }

                }

            }

        }

    }

}
