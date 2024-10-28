import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Controls.Material 2.15
import QtQuick.Layouts 1.15
import "../../Custom"

Popup {
    id: _popup

    property var _config: {
        "panorama": null
    }

    function combo_value(value) {
        value = value.toLowerCase();
        if (value === 'none')
            value = false;

        return value;
    }

    function reset() {
        if (!_config['panorama'])
            return ;

        let st = _config['panorama']['stitch'];
        // XXX 대소문자 문제로 {id}.model.indexOf({value}) 대신 array 직접 지정
        // 문제 발생 시, config.yaml의 옵션을 GUI 표기와 통일
        _perspective.currentIndex = ['panorama', 'scan'].indexOf(st['perspective']);
        _warp.currentIndex = ['plane', 'spherical'].indexOf(st['warp']);
        _compose_scale.value = st['compose_scale'] * 100;
        _warp_threshold.value = st['warp_threshold'] * 100;
        // -
        let bl = _config['panorama']['blend'];
        let blend_type = ['feather', 'multiband', 'no'];
        _ir_blend_type.currentIndex = blend_type.indexOf(bl['type']['IR']);
        _vis_blend_type.currentIndex = blend_type.indexOf(bl['type']['VIS']);
        _ir_blend_strength.value = bl['strength']['IR'] * 100;
        _vis_blend_strength.value = bl['strength']['VIS'] * 100;
        // -
        let prep = _config['panorama']['preprocess'];
        let contrast = ['equalization', 'normalization', null];
        let denoise = ['bilateral', 'gaussian', null];
        _ir_masking_threshold.value = prep['IR']['masking_threshold'] * 100;
        _ir_contrast.currentIndex = contrast.indexOf(prep['IR']['contrast']);
        _ir_denoise.currentIndex = denoise.indexOf(prep['IR']['denoise']);
        _vis_contrast.currentIndex = contrast.indexOf(prep['VIS']['contrast']);
        _vis_denoise.currentIndex = denoise.indexOf(prep['VIS']['denoise']);
    }

    function configure() {
        _config = {
            "panorama": {
                "stitch": {
                    "perspective": _perspective.currentText.toLowerCase(),
                    "warp": _warp.currentText.toLowerCase(),
                    "compose_scale": _compose_scale.value / 100,
                    "warp_threshold": _warp_threshold.value / 100
                },
                "blend": {
                    "type": {
                        "IR": combo_value(_ir_blend_type.currentText),
                        "VIS": combo_value(_vis_blend_type.currentText)
                    },
                    "strength": {
                        "IR": _ir_blend_strength.value / 100,
                        "VIS": _vis_blend_strength.value / 100
                    }
                },
                "preprocess": {
                    "IR": {
                        "contrast": combo_value(_ir_contrast.currentText),
                        "denoise": combo_value(_ir_denoise.currentText),
                        "masking_threshold": _ir_masking_threshold.value / 100
                    },
                    "VIS": {
                        "contrast": combo_value(_vis_contrast.currentText),
                        "denoise": combo_value(_vis_denoise.currentText)
                    }
                }
            }
        };
        con.configure(JSON.stringify(_config));
    }

    function update_config(config) {
        _config['panorama'] = config['panorama'];
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
            Layout.minimumWidth: 450
            Layout.maximumWidth: 750
            spacing: 20

            Label {
                id: _title

                Layout.fillWidth: true
                font.pointSize: 16
                font.weight: Font.Medium
                text: '파노라마 생성 설정'
            }

            ColumnLayout {
                spacing: 0

                Label {
                    Layout.fillWidth: true
                    font.weight: Font.Medium
                    font.pointSize: 13
                    text: '정합 설정'
                }

                GridLayout {
                    Layout.fillWidth: true
                    columns: 4
                    columnSpacing: 20

                    Label {
                        Layout.fillWidth: true
                        text: '촬영 방법'
                    }

                    ComboBox {
                        id: _perspective

                        Layout.fillWidth: true
                        model: ['Panorama', 'Scan']
                    }

                    Label {
                        Layout.fillWidth: true
                        text: '투영 방법'
                    }

                    ComboBox {
                        id: _warp

                        Layout.fillWidth: true
                        model: ['Plane', 'Spherical']
                    }

                    Label {
                        Layout.fillWidth: true
                        text: '열화상 Blend 종류'
                    }

                    ComboBox {
                        id: _ir_blend_type

                        Layout.fillWidth: true
                        model: ['Feather', 'Multiband', 'None']
                    }

                    Label {
                        Layout.fillWidth: true
                        text: '실화상 Blend 종류'
                    }

                    ComboBox {
                        id: _vis_blend_type

                        Layout.fillWidth: true
                        model: ['Feather', 'Multiband', 'None']
                    }

                    Label {
                        Layout.fillWidth: true
                        text: '열화상 Blend 강도'
                    }

                    FloatSpinBox {
                        id: _ir_blend_strength

                        value: 5
                        from: 1
                        to: 100
                        stepSize: 1
                    }

                    Label {
                        Layout.fillWidth: true
                        text: '실화상 Blend 강도'
                    }

                    FloatSpinBox {
                        id: _vis_blend_strength

                        value: 5
                        from: 1
                        to: 100
                        stepSize: 1
                    }

                    Label {
                        Layout.fillWidth: true
                        text: '스케일'
                    }

                    FloatSpinBox {
                        id: _compose_scale

                        Layout.fillWidth: true
                        value: 100
                        from: 10
                        to: 100
                        stepSize: 5
                    }

                    Label {
                        Layout.fillWidth: true
                        text: '변형 한계'
                    }

                    FloatSpinBox {
                        id: _warp_threshold

                        Layout.fillWidth: true
                        value: 2000
                        from: 100
                        to: 10000
                        stepSize: 100
                        decimals: 1
                    }

                }

            }

            ColumnLayout {
                GridLayout {
                    Layout.fillWidth: true
                    columns: 4
                    columnSpacing: 20

                    // title
                    Label {
                        Layout.columnSpan: 2
                        font.weight: Font.Medium
                        font.pointSize: 13
                        text: '열화상 전처리'
                    }

                    Label {
                        Layout.columnSpan: 2
                        font.weight: Font.Medium
                        font.pointSize: 13
                        text: '실화상 전처리'
                    }

                    // options
                    Label {
                        text: '명암 보정'
                    }

                    ComboBox {
                        id: _ir_contrast

                        Layout.fillWidth: true
                        model: ['Equalization', 'Normalization', 'None']
                    }

                    Label {
                        text: '명암 보정'
                    }

                    ComboBox {
                        id: _vis_contrast

                        Layout.fillWidth: true
                        model: ['Equalization', 'Normalization', 'None']
                    }

                    Label {
                        text: '노이즈 제거'
                    }

                    ComboBox {
                        id: _ir_denoise

                        Layout.fillWidth: true
                        model: ['Bilateral', 'Gaussian', 'None']
                    }

                    Label {
                        text: '노이즈 제거'
                    }

                    ComboBox {
                        id: _vis_denoise

                        Layout.fillWidth: true
                        model: ['Bilateral', 'Gaussian', 'None']
                    }

                }

                RowLayout {
                    Label {
                        text: '마스킹 온도'
                    }

                    FloatSpinBox {
                        id: _ir_masking_threshold

                        Layout.preferredWidth: _ir_denoise.width
                        value: -3000
                        from: -10000
                        to: 5000
                        stepSize: 100
                        decimals: 1
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
