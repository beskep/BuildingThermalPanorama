import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Controls.Material 2.15
import QtQuick.Layouts 1.15
import QtQuick.Window 2.15
import QtGraphicalEffects 1.0
import "../Custom"
import "../Button" as Btn
import "OptionPopup" as Opt
import Backend 1.0

Pane {
    function init() {
        con.rgst_reset();
        if (app.separate_panorama)
            con.rgst_pano_draw();
        else if (image_model.count)
            con.rgst_plot(image_model.get(0)['path']);
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
    objectName: 'registration_panel'

    Opt.Registration {
        id: _option
    }

    ColumnLayout {
        anchors.fill: parent

        ToolBar {
            RowLayout {
                spacing: 0

                Btn.ToolButton {
                    text: qsTr('자동 정합')
                    icon: '\ue663'
                    onReleased: con.command('register')
                    visible: !app.separate_panorama
                    ToolTip.visible: hovered
                    ToolTip.delay: 500
                    ToolTip.text: qsTr('전체 열화상·실화상 자동 정합')
                }

                Btn.ToolButton {
                    text: qsTr('저장')
                    icon: '\ue161'
                    onReleased: con.rgst_save()
                    ToolTip.visible: hovered
                    ToolTip.delay: 500
                    ToolTip.text: qsTr('대상 영상의 수동 정합 결과 저장')
                }

                ToolSeparator {
                }

                Btn.OpenFolder {
                    onReleased: con.open_dir(app.separate_panorama ? 'PANO' : 'RGST')
                }

                ToolSeparator {
                }

                Btn.Navigation {
                    index: app.separate_panorama ? 4 : 1
                }

                ToolSeparator {
                }

                Btn.Setting {
                    enabled: !app.separate_panorama
                    onReleased: _option.open()
                    ToolTip.text: '자동 열·실화상 정합 설정'
                }

                Btn.Help {
                    onReleased: con.open_help(6)
                }

            }

        }

        RowLayout {
            spacing: 10

            Pane {
                Material.elevation: 2
                Layout.fillHeight: true
                Layout.preferredWidth: 300
                padding: 5
                visible: !separate_panorama

                ListView {
                    id: image_view

                    anchors.fill: parent
                    clip: true

                    ScrollBar.vertical: ScrollBar {
                        policy: ScrollBar.AsNeeded
                    }

                    model: ListModel {
                        id: image_model
                    }

                    delegate: Pane {
                        Material.elevation: 0
                        padding: 5
                        width: image_view.width - 20
                        height: width * 3 / 4 + 10

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
                            onReleased: con.rgst_plot(path)
                            onEntered: _bc.brightness = -0.25
                            onExited: _bc.brightness = 0
                        }

                    }

                }

            }

            Pane {
                Material.elevation: 2
                Layout.fillHeight: true
                Layout.fillWidth: true
                padding: 0

                ColumnLayout {
                    anchors.fill: parent

                    Pane {
                        Layout.fillHeight: true
                        Layout.fillWidth: true
                        padding: 5

                        ColumnLayout {
                            anchors.fill: parent

                            Pane {
                                // plot 상호작용 버튼
                                Material.elevation: 1
                                padding: 0
                                leftPadding: 5

                                RowLayout {
                                    RowLayout {
                                        visible: _expand.expanded

                                        Btn.MiniToolButton {
                                            id: _point

                                            text: '지점 선택'
                                            icon: '\ue55c'
                                            down: true
                                            ToolTip.text: '열화상과 실화상의 대응되는 네 지점을 선택해서 수동으로 정합'
                                            onReleased: {
                                                down = true;
                                                _zoom.down = false;
                                            }
                                        }

                                        Btn.MiniToolButton {
                                            id: _zoom

                                            text: '확대'
                                            icon: '\ue56b'
                                            ToolTip.text: '정밀한 지점 선택을 위해 확대할 영역 지정'
                                            onDownChanged: con.rgst_zoom(down)
                                            onReleased: {
                                                down = true;
                                                _point.down = false;
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
                                                con.rgst_set_grid(down);
                                            }
                                        }

                                        ToolSeparator {
                                            leftPadding: 2
                                            rightPadding: 2
                                        }

                                        Btn.MiniToolButton {
                                            text: '초기시점'
                                            icon: '\ue88a'
                                            ToolTip.text: '영역 확대를 취소하고 전체 영상 표시'
                                            onReleased: con.rgst_home()
                                        }

                                        Btn.MiniToolButton {
                                            text: '취소'
                                            icon: '\ue14a'
                                            ToolTip.text: '수동 정합 취소'
                                            onReleased: con.rgst_reset()
                                        }

                                    }

                                    Btn.Expand {
                                        id: _expand

                                        padding: 0
                                    }

                                }

                            }

                            FigureCanvas {
                                id: plot

                                Layout.fillHeight: true
                                Layout.fillWidth: true
                                objectName: 'registration_plot'
                                dpi_ratio: Screen.devicePixelRatio
                            }

                        }

                    }

                }

            }

        }

    }

}
