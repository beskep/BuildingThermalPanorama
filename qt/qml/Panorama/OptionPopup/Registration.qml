import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Controls.Material 2.15
import QtQuick.Layouts 1.15
import "../../Custom"

Popup {
    id: _popup

    property int tooltip_width: 850
    property var _config: {
        "registration": null
    }

    function reset() {
        let cfg = _config['registration'];
        if (!cfg)
            return ;

        _hist_eq.checked = cfg['preprocess']['equalize_histogram'];
        _unsharp.checked = cfg['preprocess']['unsharp'];
        _metric.currentIndex = ['JointHistMI', 'MattesMI', 'MeanSquare'].indexOf(cfg['metric']);
        _transformation.currentIndex = ['Similarity', 'Affine'].indexOf(cfg['transformation']);
        _bins.currentIndex = ['auto', 'fd', 'sqrt'].indexOf(cfg['bins']);
        _optimizer.currentIndex = ['gradient_descent', 'powell'].indexOf(cfg['optimizer']);
    }

    function configure() {
        _config = {
            "registration": {
                "preprocess": {
                    "equalize_histogram": (_hist_eq.checkState === Qt.Checked),
                    "unsharp": (_unsharp.checkState === Qt.Checked)
                },
                "metric": _metric.currentText,
                "transformation": _transformation.currentText,
                "bins": ['auto', 'fd', 'sqrt'][_bins.currentIndex],
                "optimizer": ['gradient_descent', 'powell'][_optimizer.currentIndex]
            }
        };
        con.configure(JSON.stringify(_config));
    }

    function update_config(config) {
        _config['registration'] = config['registration'];
        reset();
    }

    anchors.centerIn: Overlay.overlay
    Material.elevation: 5
    padding: 0
    height: _content.implicitHeight

    ColumnLayout {
        id: _content

        anchors.fill: parent

        ColumnLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            Layout.margins: 20
            Layout.minimumWidth: 400
            Layout.maximumWidth: 750
            spacing: 20

            Label {
                id: _title

                Layout.fillWidth: true
                font.pointSize: 16
                font.weight: Font.Medium
                text: '자동 열·실화상 정합 설정'
            }

            ColumnLayout {
                spacing: 0

                Label {
                    Layout.fillWidth: true
                    font.weight: Font.Medium
                    font.pointSize: 13
                    text: '전처리 설정'
                }

                RowLayout {
                    CheckBox {
                        id: _hist_eq

                        text: 'HistogramEq'
                        checked: true
                        ToolTip.visible: hovered
                        ToolTip.text: '히스토그램 평활화 방법을 통해 영상의 명암 대비를 높여 밝기 및 선명도를 보강합니다.'
                    }

                    Rectangle {
                        width: 50
                    }

                    CheckBox {
                        id: _unsharp

                        text: 'Sharpening'
                        ToolTip.visible: hovered
                        ToolTip.text: '샤프닝 방법을 통해 영상의 엣지 부분의 대비를 높여 객체의 경계를 강화합니다.'
                    }

                }

            }

            ColumnLayout {
                spacing: 0

                Label {
                    Layout.fillWidth: true
                    font.weight: Font.Medium
                    font.pointSize: 13
                    text: '수치적 정합 설정'
                }

                GridLayout {
                    columns: 2
                    Layout.fillWidth: true

                    Label {
                        text: '최적화 변수'
                        Layout.fillWidth: true
                    }

                    ComboBox {
                        id: _metric

                        Layout.fillWidth: true
                        model: ['JointHistMI', 'MattesMI', 'MeanSquare']

                        ToolTip {
                            text: 'JointHistMI(기본): 두 영상의 명암분포 유사도를 측정하기 위해 조인트 히스토그램 상에서 상호의존정보를 이용합니다.<br>MattesMI: 최적화를 위한 반복마다 새로운 세트를 사용하는 대신, 픽셀 위치에 대한 단일 세트를 사용합니다.<br>MeanSquare: 두 영상의 정합 시 평균 제곱 오차를 이용하여 평가합니다.'
                            implicitWidth: tooltip_width
                            visible: parent.hovered
                        }

                    }

                    Label {
                        text: '영상 변환 방법'
                        Layout.fillWidth: true
                    }

                    ComboBox {
                        id: _transformation

                        Layout.fillWidth: true
                        model: ['Similarity', 'Affine']

                        ToolTip {
                            text: 'Similarity(기본): 평행이동, 회전, 크기 변화를 반영하여 변환합니다.<br>Affine: Similarity에 선형성을 보존하여 변환합니다.'
                            implicitWidth: tooltip_width
                            visible: parent.hovered
                        }

                    }

                    Label {
                        text: '밝기 구간 분할 방법'
                        Layout.fillWidth: true
                    }

                    ComboBox {
                        id: _bins

                        Layout.fillWidth: true
                        model: ['Auto', 'Freedman-Diaconis', 'Square Root']

                        ToolTip {
                            text: 'Auto(기본): 데이터 크기와 분산을 고려해 구간 개수를 결정합니다.<br>Freedman-Diaconis: 사분범위와 데이터 개수를 고려해 구간 개수를 결정합니다.<br>Square Root: 데이터 개수의 제곱근으로 구간 개수를 결정합니다 (다른 방법의 구간 개수가 너무 많을 때 선택).'
                            implicitWidth: tooltip_width
                            visible: parent.hovered
                        }

                    }

                    Label {
                        text: '최적화 방법'
                        Layout.fillWidth: true
                    }

                    ComboBox {
                        id: _optimizer

                        Layout.fillWidth: true
                        model: ['Gradient Descent', 'Powell']

                        ToolTip {
                            text: 'Gradient Descent(기본): 경사하강법을 통해 최적화합니다.<br>Powell:Powell\'s conjugate direction method를 통해 최적화합니다.'
                            implicitWidth: tooltip_width
                            visible: parent.hovered
                        }

                    }

                }

            }

            RowLayout {
                Layout.alignment: Qt.AlignRight | Qt.AlignBottom

                Button {
                    flat: true
                    text: 'Cancel'
                    onClicked: {
                        reset();
                        _popup.close();
                    }
                }

                Button {
                    flat: true
                    text: 'OK'
                    onClicked: {
                        configure();
                        _popup.close();
                    }
                }

            }

        }

    }

}
