<template>
    <el-scrollbar>
        <el-radio-group v-model="isCollapse" style="margin-bottom: 20px">
            <el-radio-button :value="false">expand</el-radio-button>
            <el-radio-button :value="true">collapse</el-radio-button>
        </el-radio-group>
        <el-menu
        default-active="chat"
        class="el-menu-vertical-demo"
        :collapse="isCollapse"
        @open="handleOpen"
        @close="handleClose"
        >
            <el-menu-item router="chat" index="chat">
                <el-icon><ChatRound /></el-icon>
                <template #title><router-link :to="{name: 'chat'}">对话</router-link></template>
            </el-menu-item>
            <el-sub-menu index="2">
                <template #title>
                    <el-icon><Document /></el-icon>
                    <span>文档</span>
                </template>
                <el-menu-item index="2-1"><router-link :to="{name: 'docs_question'}">文档提问</router-link></el-menu-item>
                <el-menu-item index="2-2">文档测试</el-menu-item>
                <el-menu-item index="2-3"><router-link :to="{name: 'docs_manage'}">文档管理</router-link></el-menu-item>
            </el-sub-menu>
        </el-menu>
    </el-scrollbar>
    
</template>
  
<script lang="ts" setup>
import { ref } from 'vue'
import {
    Document,
    Setting,
    ChatRound,
} from '@element-plus/icons-vue'
import { useRouter } from 'vue-router';

const router = useRouter()
router.push('/chat')  // 默认打开对话菜单

const isCollapse = ref(false)
const handleOpen = (key: string, keyPath: string[]) => {
    // console.log(key, keyPath)
}
const handleClose = (key: string, keyPath: string[]) => {
    // console.log(key, keyPath)
}
</script>

<style>
.el-menu-vertical-demo{
    height: 100%;
}
.el-menu-vertical-demo:not(.el-menu--collapse) {
    width: 200px;
    /* min-height: 400px; */
}
</style>
  

