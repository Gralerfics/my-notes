# ThinkBook 16p G6 IAX Ubuntu 22.04 音频修复记录

记录时间：2026-04-26

机器与系统：

- 机型：Lenovo ThinkBook 16p G6 IAX
- 系统：Ubuntu 22.04.5 LTS
- 内核：`6.8.0-110-generic`
- HDA Codec：Realtek ALC287
- HDA Codec Subsystem ID：`0x17aa3921`
- 功放：Cirrus Logic CS35L56，ACPI HID `CSC3556`

## 结果

最终成功状态的关键日志：

```text
cs35l56-hda i2c-CSC3556:00-cs35l56-hda.0: DSP1: cirrus/cs35l56-b0-dsp1-misc-17aa3921.wmfw
cs35l56-hda i2c-CSC3556:00-cs35l56-hda.0: DSP1: cirrus/cs35l56-b0-dsp1-misc-17aa3921-amp1.bin
snd_hda_codec_realtek ehdaudio0D0: bound i2c-CSC3556:00-cs35l56-hda.0
cs35l56-hda i2c-CSC3556:00-cs35l56-hda.1: DSP1: cirrus/cs35l56-b0-dsp1-misc-17aa3921-amp2.bin
snd_hda_codec_realtek ehdaudio0D0: bound i2c-CSC3556:00-cs35l56-hda.1
```

问题根因分两层：

1. Ubuntu 22.04 自带的 SOF 固件太旧，缺少 Arrow Lake/`arl-s` 相关固件和拓扑，导致最初无声。
2. Ubuntu 22.04 HWE `6.8.0-110` 的 Realtek HDA 驱动没有给 `17aa:3921` 这台机器绑定 `CSC3556` I2C 双 CS35L56 功放，导致虽然有声音，但低频、响度和质感很差。

## 1. 文件更改记录

### SOF 固件

来源：官方 SOF release `sof-bin-2025.12.2.tar.gz`。

下载/解压临时文件：

```text
/tmp/sof-bin-2025.12.2.tar.gz
/tmp/sof-bin-2025.12.2/
```

安装到系统的关键文件：

```text
/lib/firmware/intel/sof-ipc4/arl-s/sof-arl-s.ri -> intel-signed/sof-arl-s.ri
/lib/firmware/intel/sof-ipc4/arl-s/intel-signed/sof-arl-s.ri
/lib/firmware/intel/sof-ace-tplg/sof-hda-generic-2ch.tplg
```

校验值：

```text
505dca0b31c56801f0787f76d0f737db4f760db6cdf850f92893277f03809969  /lib/firmware/intel/sof-ipc4/arl-s/sof-arl-s.ri
0d0dbc484d788adfbe383edef946bdcabfadb7dbdcd7a86914b73c87dfb0d10d  /lib/firmware/intel/sof-ace-tplg/sof-hda-generic-2ch.tplg
```

注意：`/lib/firmware/intel/sof-ipc4/arl-s/sof-arl-s.ri` 是符号链接，实际目标是：

```text
/usr/lib/firmware/intel/sof-ipc4/arl-s/intel-signed/sof-arl-s.ri
```

### CS35L56 固件

安装到系统的文件：

```text
/lib/firmware/cirrus/cs35l56/CS35L56_Rev3.11.21.wmfw
/lib/firmware/cirrus/cs35l56-b0-dsp1-misc-17aa3921-spkid1-amp1.bin
/lib/firmware/cirrus/cs35l56-b0-dsp1-misc-17aa3921-spkid1-amp2.bin
```

为了适配当前驱动实际请求的文件名，增加了这些符号链接：

```text
/lib/firmware/cirrus/cs35l56-b0-dsp1-misc-17aa3921-spkid1.wmfw -> cs35l56/CS35L56_Rev3.11.21.wmfw
/lib/firmware/cirrus/cs35l56-b0-dsp1-misc-17aa3921.wmfw -> cs35l56/CS35L56_Rev3.11.21.wmfw
/lib/firmware/cirrus/cs35l56-b0-dsp1-misc-17aa3921-amp1.bin -> cs35l56-b0-dsp1-misc-17aa3921-spkid1-amp1.bin
/lib/firmware/cirrus/cs35l56-b0-dsp1-misc-17aa3921-amp2.bin -> cs35l56-b0-dsp1-misc-17aa3921-spkid1-amp2.bin
```

校验值：

```text
7570512b9f111cb56452f694b5dfd59bb1ca067b3e1d04935a73f7551ea7538a  /lib/firmware/cirrus/cs35l56/CS35L56_Rev3.11.21.wmfw
8ad3ac85b813a53e944a9c8171199a45380ed86d411f2e403b29bfe61cd9d8c7  /lib/firmware/cirrus/cs35l56-b0-dsp1-misc-17aa3921-spkid1-amp1.bin
8aa7af94651cfff8ea6fb55b329476aeeb37fd31b5274d179173a50eb2137757  /lib/firmware/cirrus/cs35l56-b0-dsp1-misc-17aa3921-spkid1-amp2.bin
```

### Realtek HDA 驱动模块

没有覆盖系统原始模块：

```text
/lib/modules/6.8.0-110-generic/kernel/sound/pci/hda/snd-hda-codec-realtek.ko
```

新增了优先加载的补丁模块：

```text
/lib/modules/6.8.0-110-generic/updates/sound/pci/hda/snd-hda-codec-realtek.ko
```

补丁模块信息：

```text
sha256: ee12a4d766156863d714da70c53e64c73fed06a94fb229a9932bb61563e690ec
srcversion: 28B97F1855B3C8D79283284
vermagic: 6.8.0-110-generic SMP preempt mod_unload modversions
```

补丁逻辑：

- 新增 `cs35l56_fixup_i2c_two()`
- 对 `17aa:3921` 增加 quirk：`ALC287_FIXUP_CS35L56_I2C_2`
- 让 Realtek ALC287 HDA codec 绑定两个 I2C CS35L56 功放：

```c
comp_generic_fixup(cdc, action, "i2c", "CSC3556", "-%s:00-cs35l56-hda.%d", 2);
```

临时构建目录：

```text
/tmp/ubuntu-kernel-src/
```

### 生成文件

执行过 `update-initramfs -u` 和 `depmod -a`，因此这些系统生成文件会被更新：

```text
/boot/initrd.img-6.8.0-110-generic
/lib/modules/6.8.0-110-generic/modules.*
```

这些是正常的派生文件，不是手写补丁。

### PulseAudio 软件音量 sink

CS35L56 功放绑定后，Ubuntu 22.04 的 PulseAudio/UCM 仍把右上角音量映射到 HDA `Master` 控件。这个控件不能有效控制新绑定的 CS35L56 功放路径，表现为 0 静音、非 0 接近满音量。

为当前用户新增：

```text
/home/gralerfics/.config/pulse/default.pa
```

内容逻辑：

- 包含系统默认 PulseAudio 配置：`.include /etc/pulse/default.pa`
- 加载 `module-remap-sink`
- 底层硬件 sink 固定为 100%
- 默认输出切到 `thinkbook_speaker_soft`
- 右上角音量控制 `thinkbook_speaker_soft` 的软件音量

关键配置：

```text
load-module module-remap-sink sink_name=thinkbook_speaker_soft master=alsa_output.pci-0000_80_1f.3-platform-skl_hda_dsp_generic.HiFi__hw_sofhdadsp__sink channels=2 channel_map=front-left,front-right master_channel_map=front-left,front-right remix=no sink_properties=device.description=ThinkBook_Speaker_Software_Volume
set-sink-volume alsa_output.pci-0000_80_1f.3-platform-skl_hda_dsp_generic.HiFi__hw_sofhdadsp__sink 100%
set-default-sink thinkbook_speaker_soft
set-sink-volume thinkbook_speaker_soft 40%
```

## 回滚

只回滚 Realtek 补丁模块：

```bash
sudo rm -f /lib/modules/$(uname -r)/updates/sound/pci/hda/snd-hda-codec-realtek.ko
sudo depmod -a
sudo reboot
```

同时回滚 CS35L56 固件和别名：

```bash
sudo rm -f \
  /lib/firmware/cirrus/cs35l56-b0-dsp1-misc-17aa3921.wmfw \
  /lib/firmware/cirrus/cs35l56-b0-dsp1-misc-17aa3921-spkid1.wmfw \
  /lib/firmware/cirrus/cs35l56-b0-dsp1-misc-17aa3921-amp1.bin \
  /lib/firmware/cirrus/cs35l56-b0-dsp1-misc-17aa3921-amp2.bin \
  /lib/firmware/cirrus/cs35l56-b0-dsp1-misc-17aa3921-spkid1-amp1.bin \
  /lib/firmware/cirrus/cs35l56-b0-dsp1-misc-17aa3921-spkid1-amp2.bin

sudo rm -f /lib/firmware/cirrus/cs35l56/CS35L56_Rev3.11.21.wmfw
sudo rmdir --ignore-fail-on-non-empty /lib/firmware/cirrus/cs35l56
sudo update-initramfs -u
sudo reboot
```

不建议回滚 SOF 固件，除非明确要恢复到最初无声状态。

回滚 PulseAudio 软件音量 sink：

```bash
rm -f ~/.config/pulse/default.pa
pulseaudio -k
```

## 2. 最简最佳复现流程

以下步骤适用于同样环境：ThinkBook 16p G6 IAX、Ubuntu 22.04、内核 `6.8.0-110-generic`、ALC287 codec SSID `17aa3921`。

### 1. 确认机器 ID

```bash
uname -r
cat /proc/asound/card*/codec#0 | grep -E 'Codec|Subsystem Id'
journalctl -k -b --no-pager | grep -Ei 'CSC3556|cs35l56|sof|firmware'
```

需要看到：

```text
Codec: Realtek ALC287
Subsystem Id: 0x17aa3921
CSC3556
```

### 2. 安装新版 SOF 固件

```bash
cd /tmp
wget https://github.com/thesofproject/sof-bin/releases/download/v2025.12.2/sof-bin-2025.12.2.tar.gz
tar -xf sof-bin-2025.12.2.tar.gz

sudo cp -a sof-bin-2025.12.2/lib/firmware/intel/sof-ipc4 /lib/firmware/intel/
sudo cp -a sof-bin-2025.12.2/lib/firmware/intel/sof-ipc4-lib /lib/firmware/intel/
sudo cp -a sof-bin-2025.12.2/lib/firmware/intel/sof-ace-tplg /lib/firmware/intel/

sudo update-initramfs -u
sudo reboot
```

### 3. 安装 CS35L56 固件

```bash
sudo mkdir -p /lib/firmware/cirrus/cs35l56

sudo install -m 0644 CS35L56_Rev3.11.21.wmfw \
  /lib/firmware/cirrus/cs35l56/

sudo install -m 0644 cs35l56-b0-dsp1-misc-17aa3921-spkid1-amp1.bin \
  /lib/firmware/cirrus/

sudo install -m 0644 cs35l56-b0-dsp1-misc-17aa3921-spkid1-amp2.bin \
  /lib/firmware/cirrus/

cd /lib/firmware/cirrus

sudo ln -sf cs35l56/CS35L56_Rev3.11.21.wmfw \
  cs35l56-b0-dsp1-misc-17aa3921-spkid1.wmfw

sudo ln -sf cs35l56/CS35L56_Rev3.11.21.wmfw \
  cs35l56-b0-dsp1-misc-17aa3921.wmfw

sudo ln -sf cs35l56-b0-dsp1-misc-17aa3921-spkid1-amp1.bin \
  cs35l56-b0-dsp1-misc-17aa3921-amp1.bin

sudo ln -sf cs35l56-b0-dsp1-misc-17aa3921-spkid1-amp2.bin \
  cs35l56-b0-dsp1-misc-17aa3921-amp2.bin

sudo update-initramfs -u
```

固件文件可以从 linux-firmware 上游获取。需要这三个文件：

```text
cirrus/cs35l56/CS35L56_Rev3.11.21.wmfw
cirrus/cs35l56-b0-dsp1-misc-17aa3921-spkid1-amp1.bin
cirrus/cs35l56-b0-dsp1-misc-17aa3921-spkid1-amp2.bin
```

### 4. 构建并安装 Realtek HDA 补丁模块

安装构建依赖：

```bash
sudo apt update
sudo apt install build-essential linux-headers-$(uname -r) curl zstd
```

下载 Ubuntu 当前内核源码：

```bash
mkdir -p /tmp/ubuntu-kernel-src
cd /tmp/ubuntu-kernel-src

curl -fSLO http://archive.ubuntu.com/ubuntu/pool/main/l/linux-hwe-6.8/linux-hwe-6.8_6.8.0.orig.tar.gz
tar -xzf linux-hwe-6.8_6.8.0.orig.tar.gz linux-6.8/sound/pci/hda linux-6.8/sound/hda
cd linux-6.8
```

修改 `sound/pci/hda/patch_realtek.c`：

```c
static void cs35l56_fixup_i2c_two(struct hda_codec *cdc, const struct hda_fixup *fix, int action)
{
	comp_generic_fixup(cdc, action, "i2c", "CSC3556", "-%s:00-cs35l56-hda.%d", 2);
}
```

在 fixup enum 中加入：

```c
ALC287_FIXUP_CS35L56_I2C_2,
```

在 `alc269_fixups[]` 中加入：

```c
[ALC287_FIXUP_CS35L56_I2C_2] = {
	.type = HDA_FIXUP_FUNC,
	.v.func = cs35l56_fixup_i2c_two,
},
```

在 `alc269_fixup_tbl[]` 的 Lenovo `17aa` 区域加入：

```c
SND_PCI_QUIRK(0x17aa, 0x3921, "ThinkBook 16p G6 IAX", ALC287_FIXUP_CS35L56_I2C_2),
```

构建并安装：

```bash
make -C /lib/modules/$(uname -r)/build M=$PWD/sound/pci/hda modules

sudo mkdir -p /lib/modules/$(uname -r)/updates/sound/pci/hda
sudo install -m 0644 sound/pci/hda/snd-hda-codec-realtek.ko \
  /lib/modules/$(uname -r)/updates/sound/pci/hda/

sudo depmod -a
sudo reboot
```

### 5. 验证

```bash
journalctl -k -b --no-pager | grep -Ei 'CSC3556|cs35l56|DSP1|Bound|bin file'
modinfo snd-hda-codec-realtek | head
```

成功时应看到：

```text
DSP1: cirrus/cs35l56-b0-dsp1-misc-17aa3921.wmfw
DSP1: cirrus/cs35l56-b0-dsp1-misc-17aa3921-amp1.bin
DSP1: cirrus/cs35l56-b0-dsp1-misc-17aa3921-amp2.bin
bound i2c-CSC3556:00-cs35l56-hda.0
bound i2c-CSC3556:00-cs35l56-hda.1
```

并且 `modinfo` 的文件路径应优先指向：

```text
/lib/modules/6.8.0-110-generic/updates/sound/pci/hda/snd-hda-codec-realtek.ko
```

### 6. 修复右上角音量滑条

如果出现“音量为 0 静音，非 0 接近满音量”，为当前用户创建 PulseAudio remap sink：

```bash
cat > ~/.config/pulse/default.pa <<'EOF'
.include /etc/pulse/default.pa

load-module module-remap-sink sink_name=thinkbook_speaker_soft master=alsa_output.pci-0000_80_1f.3-platform-skl_hda_dsp_generic.HiFi__hw_sofhdadsp__sink channels=2 channel_map=front-left,front-right master_channel_map=front-left,front-right remix=no sink_properties=device.description=ThinkBook_Speaker_Software_Volume
set-sink-volume alsa_output.pci-0000_80_1f.3-platform-skl_hda_dsp_generic.HiFi__hw_sofhdadsp__sink 100%
set-default-sink thinkbook_speaker_soft
set-sink-volume thinkbook_speaker_soft 40%
EOF

pulseaudio -k
```

验证：

```bash
pactl list sinks short
pactl info | grep -E '默认音频入口|Default Sink'
pactl get-sink-volume @DEFAULT_SINK@
```

默认输出应为：

```text
thinkbook_speaker_soft
```
