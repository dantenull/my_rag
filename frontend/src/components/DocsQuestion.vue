<template>
    <el-table 
        ref="multipleTableRef"
        :data="tableData" 
        style="width: 100%" 
        height="300"
        @selection-change="handleSelectionChange"
        >
        <el-table-column type="selection" width="55" />
        <el-table-column prop="upload_date" label="上传日期" width="120" />
        <el-table-column prop="name" label="名称" width="300" />
        <el-table-column prop="size" label="大小" width="120" />
        <el-table-column prop="max_page_num" label="页数" width="120" />
    </el-table>

    <el-input
        v-model="chat_records"
        type="textarea"
        :autosize="autosize"
    />

    <el-form
        ref="ruleFormRef"
        style="max-width: 600px"
        :model="ruleForm"
        status-icon
        :rules="rules"
        label-width="auto"
        class="chat-form"
    >
        <el-form-item label="" prop="pass">
            <el-input v-model="ruleForm.chat" />
        </el-form-item>
        <el-form-item>
            <el-select
                v-model="option_value"
                value-key="id"
                placeholder="Select"
                style="width: 240px"
            >
                <el-option
                    v-for="item in options"
                    :key="item.id"
                    :label="item.label"
                    :value="item"
                />
            </el-select>
        </el-form-item>
        <el-form-item>
            <el-input-number v-model="num" :min="1" :max="100" />
        </el-form-item>
        <el-form-item>
            <el-button type="primary" @click="submitForm(ruleFormRef)" class="chat-form-submit">
                Submit
            </el-button>
        </el-form-item>
    </el-form>
</template>

<script lang="ts" setup>
import { reactive, ref } from 'vue'
import http from '../http-common'
import { ElMessage, ElLoading, ElTable } from 'element-plus'
import type { FormInstance, FormRules } from 'element-plus'

interface File {
    upload_date: string
    name: string
    size: string
    max_page_num: string
}

const tableData = ref([]);
const multipleTableRef = ref<InstanceType<typeof ElTable>>()
const multipleSelection = ref<File[]>([])

const get_file_list = () => {
    http.get('/file_list')
    .then(response => {
        console.log(response.data);
        tableData.value = response.data;
    })
}
get_file_list();

const handleSelectionChange = (val: File[]) => {
    multipleSelection.value = val;
}

const autosize = { minRows: 5, maxRows: 10 }
const chat_records = ref('')

const ruleFormRef = ref<FormInstance>()
const ruleForm = reactive({
    chat: '',
})
const num = ref(1);

type Option = {
    id: number
    label: string
    url: string
    disable: boolean
}
const option_value = ref<Option>()
const options = ref([
    { id: 1, label: '生成示例答案', url: 'similarity_search', disable: false },
    { id: 2, label: '生成相关问题', url: 'similarity_search1', disable: false },
    { id: 3, label: '根据目录', url: 'query_document', disable: true },
    { id: 4, label: '通过elasticsearch', url: 'similarity_search_by_es', disable: false },
])

// 表单输入检查
const rules = reactive<FormRules<typeof ruleForm>>({
    chat: [{ trigger: 'blur' }],
})

const submitForm = (formEl: FormInstance | undefined) => {
    // if (!formEl) return
    // formEl.validate((valid) => {
    //     if (valid) {
    //         console.log('submit!')
    //     } else {
    //         console.log('error submit!')
    //         return false
    //     }
    // })
    console.log(option_value.value)
    if (!option_value.value){
        ElMessage({
            message: '请选择方法',
            type: 'warning',
            plain: true,
        });
        return;
    }
    if (option_value.value.disable){
        ElMessage({
            message: '此方法不可用',
            type: 'warning',
            plain: true,
        });
        return;
    }
    if (!ruleForm.chat){
        ElMessage({
            message: '请输入问题',
            type: 'warning',
            plain: true,
        });
        return;
    }
    if (multipleSelection.value.length != 1){
        ElMessage({
            message: '请选择一个文件',
            type: 'warning',
            plain: true,
        });
        return;
    }
    http.post(option_value.value.url, {'query': ruleForm.chat, 'n': num.value, 'file_name': multipleSelection.value[0].name})
    .then(response => {
        console.log(response.data);
        chat_records.value = response.data;
    })
}
</script>

<style>
.chat-form{
    margin-top: 20px;
}
.chat-form-submit{
    float: right;
}
</style>
