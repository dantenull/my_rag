<template>
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
            <el-button type="primary" @click="submitForm(ruleFormRef)" class="chat-form-submit">
                Submit
            </el-button>
        </el-form-item>
    </el-form>
</template>

<script lang="ts" setup>
import { reactive, ref } from 'vue'
// import { useRouter } from 'vue-router';
import type { FormInstance, FormRules } from 'element-plus'
import http from '../http-common'

// const router = useRouter()
// router.push('/chat')

const autosize = { minRows: 5, maxRows: 10 }
const chat_records = ref('')

const ruleFormRef = ref<FormInstance>()
const ruleForm = reactive({
    chat: '',
})

// 表单输入检查
const rules = reactive<FormRules<typeof ruleForm>>({
    chat: [{ trigger: 'blur' }],
})

const submitForm = (formEl: FormInstance | undefined) => {
    if (!formEl) return
    formEl.validate((valid) => {
        console.log(valid)
        if (valid) {
            console.log('submit!')
        } else {
            console.log('error submit!')
            return false
        }
    })
    http.post('/chat', {'message': ruleForm.chat})
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
