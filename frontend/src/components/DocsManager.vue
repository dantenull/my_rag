<template>
    <!-- <el-upload
        v-model:file-list="fileList"
        class="upload"
        :on-change="handleChange"
        :before-upload="beforeUpload"
        :auto-upload="false"
        limit="1"
    >
        <template #trigger>
            <el-button type="primary">select file</el-button>
        </template>
        <el-button class="ml-3" type="success" @click="submitUpload">
            upload to server
        </el-button>
        <template #tip>
            <div class="el-upload__tip">
                files with a size less than 100MB
            </div>
        </template>
    </el-upload> -->

    <div>
        <!-- <el-button type="primary">上传文件</el-button> -->
        <input type="file" @change="handleFileChange" />
    </div>
    <el-table 
        :data="tableData" 
        style="width: 100%" 
        height="500"
        >
        <el-table-column type="selection" width="50" />
        <el-table-column prop="upload_date" label="上传日期" width="100" />
        <el-table-column prop="name" label="名称" width="200" />
        <el-table-column prop="size" label="大小" width="100" />
        <el-table-column prop="max_page_num" label="页数" width="100" />
        <el-table-column prop="upload_state" label="上传状态" width="100" />
        <el-table-column fixed="right" label="操作" width="100">
            <template #default="scope">
                <el-button link type="primary" size="small" @click.prevent="deleteFile(scope.$index)">
                    删除
                </el-button>
            </template>
        </el-table-column>
    </el-table>
</template>

<script lang="ts" setup>
import { reactive, ref } from 'vue'
// import type { UploadInstance, UploadProps, UploadUserFile } from 'element-plus'
import settings from '../settings';
import http from '../http-common'
import { ElMessage, ElLoading } from 'element-plus'

const tableData = ref([]);

const get_file_list = () => {
    http.get('/file_list')
    .then(response => {
        console.log(response.data);
        tableData.value = response.data;
    })
}
get_file_list();

const handleFileChange = async (event) => {
    const file = event.target.files[0];
    const action_url = settings.baseURL + 'upload';
    const headers = {
        'Content-Type': 'multipart/form-data', 
        'accept': 'application/json',
    };
    console.log(file)
    let formData = new FormData();
    formData.append('file', file);
    const loading = ElLoading.service({
        lock: true,
        text: 'Loading',
        background: 'rgba(0, 0, 0, 0.7)',
    })
    http({
        method: 'post', 
        url: action_url,
        data: formData,
        headers: headers,
    })
    .then(
        response => { 
            loading.close();
            
            let result = response.data;
            console.log(result);
            for (let task of result){
                let task_id = task['task_id'];
                if (task_id){
                    checkStatusWs(task_id);
                }
            }
            ElMessage({
                message: '上传成功',
                type: 'success',
                plain: true,
            });
            get_file_list();
        }, 
        error => {
            loading.close();
            console.log(error.data);
            ElMessage({
                message: '上传失败',
                type: 'error',
                plain: true,
            });
        }
    )
}

const deleteFile = (index: number) => {
    http.post('/delete_by_file', {'file_id': tableData.value[index].file_id})
    .then(response => {
        get_file_list();
    })
}

async function checkStatus(taskId) {
    console.log('checkStatus');
    http.post('/get_celery_task_status', {'task_id': taskId}).then(response => {
        let result = response.data;
        if (result.status === "Pending") {
            setTimeout(() => checkStatus(taskId), 1000);  // 轮询
        } else if (result.status === "Completed") {
            get_file_list();
        }
    })
}

function checkStatusWs(taskId) {
    console.log('checkStatusWs');
    const websocket = new WebSocket('ws://localhost:8008/ws/get_celery_task_status');
    websocket.onopen = (event) => {
        websocket.send(JSON.stringify({ task_id: taskId }));
    };
    websocket.onmessage = (event) => {
        console.log(event.data);
    }
    websocket.onclose = (event) => {
        get_file_list();
    }
}

// const handleChange: UploadProps['onChange'] = (uploadFile, uploadFiles) => {
//     console.log(uploadFile);
//     file.value = uploadFile;
//     // fileList.value.push(uploadFile);
//     // UploadFileInfo.value = {
//     //     'file': uploadRef.value,
//     //     'filename': uploadFile.raw.name,
//     //     'size': uploadFile.size,
//     //     'headers': {},
//     //     'content_type': uploadFile.raw.type,
//     // }
//     // UploadFileInfo.value!.submit()
// }
// const beforeUpload: UploadProps['beforeUpload'] = (rawFile) => {
//     console.log(rawFile);
//     if (rawFile.size > 100 * 1024 * 1024) {
//         return false;
//     } 
// }
// const submitUpload = () => {
//     const action_url = settings.baseURL + 'upload';
//     const headers = {
//         'Content-Type': 'multipart/form-data', 
//         'accept': 'application/json',
//     };
//     let data = {
//         // 'file': file.value.raw,
//         'filename': file.value.name,
//         'size': file.value.size,
//         'headers': file.value.headers,
//         'content_type': file.value.content_type,
//     }
//     // const upload = ref<UploadInstance>()
//     // console.log(upload.value);
//     const jsonStr = JSON.stringify(data);
//     const blob = new Blob([jsonStr], {
//         type: 'application/json'
//     })
//     let formData = new FormData();
//     // formData.append('notes', blob);
//     console.log(file.value.raw)
//     formData.append('fileb', file.value);
//     // formData.append('filename', file.value.name);
//     // formData.append('size', file.value.size);
//     // formData.append('headers', file.value.headers);
//     // formData.append('content_type', file.value.content_type);
    
//     // console.log(formData);
//     // console.log(file.value);
//     http({
//         method: 'post', 
//         url: action_url,
//         data: formData,
//         headers: headers,
//     })
//     .then(response => {
//         console.log(response.data);
//     })
// }
</script>