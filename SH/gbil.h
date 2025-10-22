
#ifndef GBIL_H
#define GBIL_H

#define IMPORTint extern "C" _declspec(dllimport) int __stdcall 
#define IMPORTchar extern "C" _declspec(dllimport) char* __stdcall 


/***************** 设备接口 ******************/
// 打开设备（根据设备序号）
IMPORTint opendevice(int index = 0);
// 打开设备（根据设备ID）
IMPORTint opendevicebyid(int vid, int pid);
// 打开设备（根据设备路径）
IMPORTint opendevicebypath(const char *path);
// 检查设备是否连接
IMPORTint isconnected(void);
// 关闭设备
IMPORTint closedevice(void);
// 复位设备
IMPORTint resetdevice(void);
// 获取设备型号
IMPORTchar getmodel(void);
// 获取设备序列号
IMPORTchar getserialnumber(void);
// 获取设备生产日期
IMPORTchar getproductiondate(void);
// 获取设备固件版本号
IMPORTchar getfirmwareversion(void);


/***************** 键盘接口 ******************/
// 按下键
IMPORTint presskeybyname(const char *key);
IMPORTint presskeybyvalue(int key);
// 释放键
IMPORTint releasekeybyname(const char *key);
IMPORTint releasekeybyvalue(int key);
// 按下并释放键
IMPORTint pressandreleasekeybyname(const char *key);
IMPORTint pressandreleasekeybyvalue(int key);
// 判断键盘按键状态
IMPORTint iskeypressedbyname(const char *key);
IMPORTint iskeypressedbyvalue(int key);
// 释放所有键盘按键
IMPORTint releaseallkey(void);
// 输入字符串
IMPORTint inputstring(const char *str);
// 获取大写锁定状态
IMPORTint getcapslock(void);
// 获取数字键盘锁定状态
IMPORTint getnumlock(void);
// 设置是否区分大小写
IMPORTint setcasesensitive(int cs);
// 设置按键延时
IMPORTint setpresskeydelay(int mind, int maxd);
// 设置输入字符串间隔时间
IMPORTint setinputstringintervaltime(int mind, int maxd);


/***************** 鼠标接口 ******************/
// 按下鼠标键
IMPORTint pressmousebutton(int mbtn);
// 释放鼠标键
IMPORTint releasemousebutton(int mbtn);
// 按下并释放鼠标键
IMPORTint pressandreleasemousebutton(int mbtn);
// 判断鼠标按键状态
IMPORTint ismousebuttonpressed(int mbtn);
// 释放所有鼠标按键
IMPORTint releaseallmousebutton(void);
// 相对移动鼠标
IMPORTint movemouserelative(int x, int y);
// 移动鼠标到指定坐标
IMPORTint movemouseto(int x, int y);
// 获取鼠标当前位置
IMPORTint getmousex(void);
IMPORTint getmousey(void);
// 移动鼠标滚轮
IMPORTint movemousewheel(int z);
// 设置鼠标按键延时
IMPORTint setpressmousebuttondelay(int mind, int maxd);
// 设置鼠标移动延时
IMPORTint setmousemovementdelay(int mind, int maxd);
// 设置鼠标移动速度
IMPORTint setmousemovementspeed(int speedvalue);
// 设置鼠标移动模式
IMPORTint setmousemovementmode(int modevalue);


/***************** 双机互联接口 ******************/
// 设置鼠标当前位置（针对不支持绝对值的鼠标）
IMPORTint setmouseposition(int x, int y);
// 设置鼠标绝对位置（针对支持绝对值的鼠标）
IMPORTint setmouseabsoluteposition(int x, int y);
// 设置被控端屏幕分辨率
IMPORTint setclientscreenresolution(int width, int height);
// 获取被控端屏幕分辨率
IMPORTchar getclientscreenresolution(void);
// 选择客户端
IMPORTint selectclient(int addr);


/***************** 加密狗接口 ******************/
// 初始化加密狗
IMPORTint initializedongle(void);
// 设置读密码
IMPORTint setreadpassword(const char* writepwd, const char* newpwd);
// 设置写密码
IMPORTint setwritepassword(const char* oldpwd, const char* newpwd);
// 从设备读字符串
IMPORTchar readstring(const char* readpwd, int addr, int count);
// 将字符串写入设备
IMPORTint writestring(const char* writepwd, const char* str, int addr);
// 设置密钥
IMPORTint setcipher(const char* writepwd, const char* cipher);
// 加密字符串
IMPORTchar encryptstring(const char* str);
// 解密字符串
IMPORTchar decryptstring(const char* str);


/***************** 电源控制接口 ******************/
// 按下电源按钮
IMPORTint presspowerbutton(void);
// 释放电源按钮
IMPORTint releasepowerbutton(void);
// 按下并释放电源按钮
IMPORTint pressandreleasepowerbutton(void);
// 获取电源工作状态
IMPORTint getpowerstatus(void);


/***************** 设备定义接口 ******************/
// 修改设备速度
IMPORTint setspeed(int speed);
// 修改设备ID
IMPORTint setdeviceid(int vid, int pid);
// 恢复设备默认ID
IMPORTint restoredeviceid(void);
// 设置产品名称
IMPORTint setproductname(const char *pn);
// 获取产品名称
IMPORTchar getproductionname(void);


#endif // !GBIL_H

